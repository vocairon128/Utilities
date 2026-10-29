import sys
import numpy as np
import pandas as pd
from scipy.signal import get_window, find_peaks

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QLabel, QComboBox, QSpinBox, QDoubleSpinBox,
    QCheckBox, QGroupBox, QGridLayout, QMessageBox, QTextEdit
)
from PyQt6.QtCore import Qt
import pyqtgraph as pg


class FFTAnalyzerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Oscilloscope CSV FFT Analyzer")
        self.resize(1250, 850)

        self.df = None
        self.time_col = None
        self.data_cols = []

        self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # ------------------- 왼쪽: 설정 패널 -------------------
        control_panel = QWidget()
        control_panel.setFixedWidth(340)
        control_layout = QVBoxLayout(control_panel)

        # 1. 파일 불러오기
        self.btn_load = QPushButton("📁 CSV 파일 불러오기")
        self.btn_load.clicked.connect(self.load_csv)
        control_layout.addWidget(self.btn_load)

        self.lbl_file_info = QLabel("선택된 파일 없음")
        self.lbl_file_info.setWordWrap(True)
        control_layout.addWidget(self.lbl_file_info)

        # 2. 데이터 및 샘플링 설정 그룹
        data_group = QGroupBox("데이터 및 샘플링 설정")
        data_grid = QGridLayout()

        data_grid.addWidget(QLabel("채널 선택:"), 0, 0)
        self.combo_channel = QComboBox()
        self.combo_channel.currentIndexChanged.connect(self.run_analysis)
        data_grid.addWidget(self.combo_channel, 0, 1)

        data_grid.addWidget(QLabel("샘플링 Rate (Hz):"), 1, 0)
        self.spin_fs = QDoubleSpinBox()
        self.spin_fs.setRange(1, 1e10)
        self.spin_fs.setValue(1000000)  # 기본값 1 MS/s
        self.spin_fs.setSingleStep(1000)
        self.spin_fs.valueChanged.connect(self.run_analysis)
        data_grid.addWidget(self.spin_fs, 1, 1)

        self.chk_auto_fs = QCheckBox("Time 열에서 $f_s$ 자동 추출")
        self.chk_auto_fs.setChecked(True)
        self.chk_auto_fs.toggled.connect(self.toggle_auto_fs)
        data_grid.addWidget(self.chk_auto_fs, 2, 0, 1, 2)

        data_group.setLayout(data_grid)
        control_layout.addWidget(data_group)

        # 3. FFT 파라미터 그룹
        fft_group = QGroupBox("FFT 제어 변수")
        fft_grid = QGridLayout()

        fft_grid.addWidget(QLabel("FFT Size (N):"), 0, 0)
        self.combo_fft_size = QComboBox()
        self.combo_fft_size.addItems(["Auto (전체 데이터)", "1024", "2048", "4096", "8192", "16384", "32768", "65536"])
        self.combo_fft_size.currentIndexChanged.connect(self.run_analysis)
        fft_grid.addWidget(self.combo_fft_size, 0, 1)

        fft_grid.addWidget(QLabel("윈도우 함수:"), 1, 0)
        self.combo_window = QComboBox()
        self.combo_window.addItems(["hann", "hamming", "flattop", "boxcar (None)"])
        self.combo_window.currentIndexChanged.connect(self.run_analysis)
        fft_grid.addWidget(self.combo_window, 1, 1)

        fft_grid.addWidget(QLabel("스펙트럼 단위:"), 2, 0)
        self.combo_unit = QComboBox()
        self.combo_unit.addItems(["Linear (Vrms)", "dBV", "PSD (V^2/Hz)"])
        self.combo_unit.currentIndexChanged.connect(self.run_analysis)
        fft_grid.addWidget(self.combo_unit, 2, 1)

        self.chk_remove_dc = QCheckBox("DC Offset 제거 (Mean Subtraction)")
        self.chk_remove_dc.setChecked(True)
        self.chk_remove_dc.toggled.connect(self.run_analysis)
        fft_grid.addWidget(self.chk_remove_dc, 3, 0, 1, 2)

        fft_group.setLayout(fft_grid)
        control_layout.addWidget(fft_group)

        # 4. 시간/주파수 범위 제어 그룹 (X축)
        range_group = QGroupBox("X축 범위 제어 (시간/주파수)")
        range_grid = QGridLayout()

        range_grid.addWidget(QLabel("시작 시간 (s):"), 0, 0)
        self.spin_t_start = QDoubleSpinBox()
        self.spin_t_start.setRange(-1e6, 1e6)
        self.spin_t_start.setDecimals(6)
        self.spin_t_start.valueChanged.connect(self.run_analysis)
        range_grid.addWidget(self.spin_t_start, 0, 1)

        range_grid.addWidget(QLabel("종료 시간 (s):"), 1, 0)
        self.spin_t_end = QDoubleSpinBox()
        self.spin_t_end.setRange(-1e6, 1e6)
        self.spin_t_end.setDecimals(6)
        self.spin_t_end.valueChanged.connect(self.run_analysis)
        range_grid.addWidget(self.spin_t_end, 1, 1)

        range_grid.addWidget(QLabel("최소 주파수 (Hz):"), 2, 0)
        self.spin_f_min = QDoubleSpinBox()
        self.spin_f_min.setRange(0, 1e10)
        self.spin_f_min.setValue(0)
        self.spin_f_min.setSingleStep(100)
        self.spin_f_min.valueChanged.connect(self.run_analysis)
        range_grid.addWidget(self.spin_f_min, 2, 1)

        range_grid.addWidget(QLabel("최대 주파수 (Hz):"), 3, 0)
        self.spin_f_max = QDoubleSpinBox()
        self.spin_f_max.setRange(0, 1e10)
        self.spin_f_max.setValue(500000)
        self.spin_f_max.setSingleStep(1000)
        self.spin_f_max.valueChanged.connect(self.run_analysis)
        range_grid.addWidget(self.spin_f_max, 3, 1)

        self.chk_auto_f_max = QCheckBox("최대 주파수 자동 (Nyquist)")
        self.chk_auto_f_max.setChecked(True)
        self.chk_auto_f_max.toggled.connect(self.toggle_auto_f_max)
        range_grid.addWidget(self.chk_auto_f_max, 4, 0, 1, 2)

        range_group.setLayout(range_grid)
        control_layout.addWidget(range_group)

        # 5. ★ Y축 스케일 제어 그룹 (신규 추가)
        y_group = QGroupBox("Y축 스케일 제어 (Voltage & Mag)")
        y_grid = QGridLayout()

        # Voltage (시간축 Y축)
        y_grid.addWidget(QLabel("Min Volt (V):"), 0, 0)
        self.spin_v_min = QDoubleSpinBox()
        self.spin_v_min.setRange(-1e5, 1e5)
        self.spin_v_min.setValue(-10.0)
        self.spin_v_min.setSingleStep(0.5)
        self.spin_v_min.setEnabled(False)
        self.spin_v_min.valueChanged.connect(self.run_analysis)
        y_grid.addWidget(self.spin_v_min, 0, 1)

        y_grid.addWidget(QLabel("Max Volt (V):"), 1, 0)
        self.spin_v_max = QDoubleSpinBox()
        self.spin_v_max.setRange(-1e5, 1e5)
        self.spin_v_max.setValue(10.0)
        self.spin_v_max.setSingleStep(0.5)
        self.spin_v_max.setEnabled(False)
        self.spin_v_max.valueChanged.connect(self.run_analysis)
        y_grid.addWidget(self.spin_v_max, 1, 1)

        self.chk_auto_y_time = QCheckBox("Volt Y축 자동 스케일")
        self.chk_auto_y_time.setChecked(True)
        self.chk_auto_y_time.toggled.connect(self.toggle_y_time_auto)
        y_grid.addWidget(self.chk_auto_y_time, 2, 0, 1, 2)

        # Magnitude (주파수축 Y축)
        y_grid.addWidget(QLabel("Min Mag (Vrms / dBV):"), 3, 0)
        self.spin_m_min = QDoubleSpinBox()
        self.spin_m_min.setRange(-1e5, 1e5)
        self.spin_m_min.setDecimals(6)  # ★ 소수점 6자리까지 표기 가능 (0.000001)
        self.spin_m_min.setValue(0.0)
        self.spin_m_min.setSingleStep(0.001)  # ★ 화살표 클릭 시 0.001 (1mV) 단위로 변경
        self.spin_m_min.setEnabled(False)
        self.spin_m_min.valueChanged.connect(self.run_analysis)
        y_grid.addWidget(self.spin_m_min, 3, 1)

        y_grid.addWidget(QLabel("Max Mag (Vrms / dBV):"), 4, 0)
        self.spin_m_max = QDoubleSpinBox()
        self.spin_m_max.setRange(-1e5, 1e5)
        self.spin_m_max.setDecimals(6)  # ★ 소수점 6자리까지 표기 가능
        self.spin_m_max.setValue(1.0)  # 기본값 1.0 Vrms
        self.spin_m_max.setSingleStep(0.001)  # ★ 0.001 (1mV) 단위 변경
        self.spin_m_max.setEnabled(False)
        self.spin_m_max.valueChanged.connect(self.run_analysis)
        y_grid.addWidget(self.spin_m_max, 4, 1)

        self.chk_auto_y_freq = QCheckBox("Magnitude Y축 자동 스케일")
        self.chk_auto_y_freq.setChecked(True)
        self.chk_auto_y_freq.toggled.connect(self.toggle_y_freq_auto)
        y_grid.addWidget(self.chk_auto_y_freq, 5, 0, 1, 2)

        y_group.setLayout(y_grid)
        control_layout.addWidget(y_group)

        # 6. 분석 결과 정보 출력창
        self.txt_info = QTextEdit()
        self.txt_info.setReadOnly(True)
        control_layout.addWidget(self.txt_info)

        control_layout.addStretch()
        main_layout.addWidget(control_panel)

        # ------------------- 오른쪽: 그래프 패널 -------------------
        graph_widget = pg.GraphicsLayoutWidget()
        main_layout.addWidget(graph_widget)

        # 시간 영역 그래프
        self.plot_time = graph_widget.addPlot(title="Time Domain Signal")
        self.plot_time.setLabel('bottom', "Time", units='s')
        self.plot_time.setLabel('left', "Voltage", units='V')
        self.plot_time.showGrid(x=True, y=True)
        self.curve_time = self.plot_time.plot(pen='c')

        graph_widget.nextRow()

        # 주파수 영역 그래프
        self.plot_freq = graph_widget.addPlot(title="Frequency Domain Spectrum (FFT)")
        self.plot_freq.setLabel('bottom', "Frequency", units='Hz')
        self.plot_freq.setLabel('left', "Magnitude")
        self.plot_freq.showGrid(x=True, y=True)
        self.curve_freq = self.plot_freq.plot(pen='m')

        # 피크 표시 마커
        self.peak_scatter = pg.ScatterPlotItem(size=10, brush=pg.mkBrush(255, 0, 0, 255), symbol='o')
        self.plot_freq.addItem(self.peak_scatter)

    def toggle_auto_fs(self, checked):
        self.spin_fs.setEnabled(not checked)
        if checked and self.df is not None:
            self.calculate_auto_fs()
            self.run_analysis()

    def toggle_auto_f_max(self, checked):
        self.spin_f_max.setEnabled(not checked)
        if checked and self.df is not None:
            fs = self.spin_fs.value()
            self.spin_f_max.setValue(fs / 2.0)
        self.run_analysis()

    def toggle_y_time_auto(self, checked):
        self.spin_v_min.setEnabled(not checked)
        self.spin_v_max.setEnabled(not checked)
        if checked:
            self.plot_time.enableAutoRange(axis=pg.ViewBox.YAxis)
        self.run_analysis()

    def toggle_y_freq_auto(self, checked):
        self.spin_m_min.setEnabled(not checked)
        self.spin_m_max.setEnabled(not checked)
        if checked:
            self.plot_freq.enableAutoRange(axis=pg.ViewBox.YAxis)
        self.run_analysis()

    def load_csv(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "CSV 파일 선택", "", "CSV Files (*.csv);;All Files (*)")
        if not file_path:
            return

        try:
            self.df = pd.read_csv(file_path, comment='#')
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) < 2:
                QMessageBox.critical(self, "오류", "CSV에 최소 2개 이상의 수치 데이터 열이 필요합니다.")
                return

            self.time_col = numeric_cols[0]
            self.data_cols = numeric_cols[1:]

            self.combo_channel.blockSignals(True)
            self.combo_channel.clear()
            self.combo_channel.addItems(self.data_cols)
            self.combo_channel.blockSignals(False)

            t_min = self.df[self.time_col].min()
            t_max = self.df[self.time_col].max()

            self.spin_t_start.blockSignals(True)
            self.spin_t_end.blockSignals(True)
            self.spin_t_start.setValue(t_min)
            self.spin_t_end.setValue(t_max)
            self.spin_t_start.blockSignals(False)
            self.spin_t_end.blockSignals(False)

            self.lbl_file_info.setText(f"파일: {file_path.split('/')[-1]}\n데이터 개수: {len(self.df):,} 개")

            if self.chk_auto_fs.isChecked():
                self.calculate_auto_fs()

            if self.chk_auto_f_max.isChecked():
                self.spin_f_max.setValue(self.spin_fs.value() / 2.0)

            self.run_analysis()

        except Exception as e:
            QMessageBox.critical(self, "오류", f"CSV 파일 로드 중 오류 발생:\n{str(e)}")

    def calculate_auto_fs(self):
        if self.df is not None and self.time_col is not None:
            t = self.df[self.time_col].values
            dt = np.mean(np.diff(t))
            if dt > 0:
                fs = 1.0 / dt
                self.spin_fs.blockSignals(True)
                self.spin_fs.setValue(fs)
                self.spin_fs.blockSignals(False)

    def run_analysis(self):
        if self.df is None or not self.combo_channel.currentText():
            return

        ch_name = self.combo_channel.currentText()
        fs = self.spin_fs.value()
        t_start = self.spin_t_start.value()
        t_end = self.spin_t_end.value()

        # 분석 구간 자르기
        sub_df = self.df[(self.df[self.time_col] >= t_start) & (self.df[self.time_col] <= t_end)]
        if len(sub_df) < 8:
            return

        t = sub_df[self.time_col].values
        signal = sub_df[ch_name].values

        if self.chk_remove_dc.isChecked():
            signal = signal - np.mean(signal)

        # 시간 영역 그래프 처리 및 Y축 제어
        self.curve_time.setData(t, signal)
        if not self.chk_auto_y_time.isChecked():
            self.plot_time.setYRange(self.spin_v_min.value(), self.spin_v_max.value(), padding=0)

        # FFT 연산
        fft_size_str = self.combo_fft_size.currentText()
        if "Auto" in fft_size_str:
            N = len(signal)
        else:
            N = int(fft_size_str)
            if N > len(signal):
                signal = np.pad(signal, (0, N - len(signal)), 'constant')
            else:
                signal = signal[:N]

        win_type = self.combo_window.currentText()
        win = get_window(win_type, len(signal))
        signal_win = signal * win
        win_norm = np.sum(win)

        fft_raw = np.fft.rfft(signal_win)
        freqs = np.fft.rfftfreq(len(signal_win), d=1.0 / fs)

        unit = self.combo_unit.currentText()
        mag = (np.abs(fft_raw) * np.sqrt(2)) / win_norm
        mag[0] = mag[0] / np.sqrt(2)

        if "dBV" in unit:
            y_data = 20 * np.log10(mag + 1e-12)
            self.plot_freq.setLabel('left', "Magnitude", units='dBV')
        elif "PSD" in unit:
            y_data = (np.abs(fft_raw) ** 2) / (fs * np.sum(win ** 2))
            self.plot_freq.setLabel('left', "PSD", units='V^2/Hz')
        else:
            y_data = mag
            self.plot_freq.setLabel('left', "Magnitude", units='Vrms')

        # X축 범위 제한 (Min/Max Filtering)
        f_min = self.spin_f_min.value()
        f_max = self.spin_f_max.value()

        freq_mask = (freqs >= f_min) & (freqs <= f_max)
        freqs_view = freqs[freq_mask]
        y_data_view = y_data[freq_mask]

        # FFT 그래프 업데이트 및 X/Y축 범위 적용
        self.curve_freq.setData(freqs_view, y_data_view)
        self.plot_freq.setXRange(f_min, f_max, padding=0)

        if not self.chk_auto_y_freq.isChecked():
            self.plot_freq.setYRange(self.spin_m_min.value(), self.spin_m_max.value(), padding=0)

        # 주요 피크 탐지
        df_res = fs / len(signal_win)
        peaks, _ = find_peaks(y_data_view, distance=max(1, int(10 / df_res)))

        info_text = f"<b>[분석 결과 정보]</b><br>"
        info_text += f"샘플링 주파수 ($f_s$): {fs / 1e3:.2f} kHz<br>"
        info_text += f"FFT 샘플 수 ($N$): {len(signal_win):,} pts<br>"
        info_text += f"주파수 분해능 ($\Delta f$): {df_res:.2f} Hz<br>"
        info_text += f"표시 범주: {f_min:.0f} Hz ~ {f_max / 1e3:.2f} kHz<br><br>"

        if len(peaks) > 0:
            sorted_peaks = sorted(peaks, key=lambda x: y_data_view[x], reverse=True)[:3]
            peak_freqs = freqs_view[sorted_peaks]
            peak_mags = y_data_view[sorted_peaks]

            self.peak_scatter.setData(x=peak_freqs, y=peak_mags)

            info_text += f"<b>[설정 범위 내 주요 Peak Top 3]</b><br>"
            for i, (pf, pm) in enumerate(zip(peak_freqs, peak_mags), 1):
                info_text += f"#{i}: <b>{pf / 1e3:.3f} kHz</b> ({pm:.3f})<br>"
        else:
            self.peak_scatter.clear()

        self.txt_info.setHtml(info_text)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FFTAnalyzerApp()
    window.show()
    sys.exit(app.exec())