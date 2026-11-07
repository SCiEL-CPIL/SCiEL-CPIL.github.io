"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.linalg import expm

_DEADLINE = 325.0
_START_TIME = time.time()

def _time_remaining():
    return _DEADLINE - (time.time() - _START_TIME)

# ---- fixed hardware constants ----
_N = 80
_DT = 0.5
_T = _N * _DT  # 40 ns
_ANH = -2.0 * np.pi * 0.25   # anharmonicity
_AMP = 0.4
_SLEW = 0.12
_DET_GRID = np.array([-2*np.pi*0.002, 0.0, 2*np.pi*0.002])
_SCALE_GRID = np.array([0.97, 1.0, 1.03])
_T1 = 25000.0
_TPHI = 40000.0

# a = diag(sqrt([1,2]),1); a_dag = a^H
_A = np.diag([1.0, np.sqrt(2.0), 1.0])
_ADAG = _A.conj().T
_NOP = _ADAG @ _A

# Lindblad operators
_L1 = _A / np.sqrt(_T1)
_L2 = _NOP * np.sqrt(2.0 / _TPHI)
_L1ADAG = _L1.conj().T
_L2ADAG = _L2.conj().T

# Precompute static parts
_L1ADAG_L1 = _L1ADAG @ _L1
_L2ADAG_L2 = _L2ADAG @ _L2
_DECAY = 0.5 * (_L1ADAG_L1 + _L2ADAG_L2)

# States for fidelity (3D, padded with 0 for |2>)
# Inputs
_S0 = np.array([1.0, 0.0, 0.0], dtype=complex)
_S1 = np.array([0.0, 1.0, 0.0], dtype=complex)
_Sp = (np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)).astype(complex)
_Sm = (np.array([1.0, -1.0, 0.0]) / np.sqrt(2.0)).astype(complex)
_Spi = (np.array([1.0, 1j, 0.0]) / np.sqrt(2.0)).astype(complex)
_Smi = (np.array([1.0, -1j, 0.0]) / np.sqrt(2.0)).astype(complex)
_INIT_STATES = [_S0, _S1, _Sp, _Sm, _Spi, _Smi]

# Targets (X gate applied to inputs)
_T0 = np.array([0.0, 1.0, 0.0], dtype=complex)
_T1 = np.array([1.0, 0.0, 0.0], dtype=complex)
_Tp = (np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)).astype(complex)
_Tm = (np.array([-1.0, 1.0, 0.0]) / np.sqrt(2.0)).astype(complex)
_Tpi = (np.array([1j, 1.0, 0.0]) / np.sqrt(2.0)).astype(complex)
_Tmi = (np.array([-1j, 1.0, 0.0]) / np.sqrt(2.0)).astype(complex)
_TARGET_STATES = [_T0, _T1, _Tp, _Tm, _Tpi, _Tmi]

def _make_pulse(alpha, amp, sigma=None, width_factor=1.0):
    """Gaussian I-channel, DRAG Q = alpha * dI/dt. Area of I = pi."""
    t = np.arange(_N) * _DT
    if sigma is None:
        sigma = 40.0 / 6.0
    # Gaussian shape
    g = np.exp(-(t - _T/2.0)**2 / (2.0 * sigma**2))
    # normalize so that sum(I*dt) = pi (area)
    area = np.sum(g) * _DT
    I = amp * g * (np.pi / area)
    # dI/dt via central differences
    dIdt = np.zeros(_N)
    dIdt[1:-1] = (I[2:] - I[:-2]) / (2.0 * _DT)
    dIdt[0] = (I[1] - I[0]) / _DT
    dIdt[-1] = (I[-1] - I[-2]) / _DT
    Q = alpha * dIdt
    return np.column_stack([I, Q])

def _valid(pulse_arr):
    # amplitude bound
    r = np.sqrt(np.sum(pulse_arr**2, axis=1))
    if np.max(r) > _AMP + 1e-9:
        return False
    # slew bound including from zero
    p0 = np.zeros(2)
    for i in range(_N):
        if np.linalg.norm(pulse_arr[i] - p0) / _DT > _SLEW + 1e-9:
            return False
        p0 = pulse_arr[i]
    if np.linalg.norm(p0) / _DT > _SLEW + 1e-9:
        return False
    return True

def _propagate(pulse_arr, det, scale):
    """Return 3x3 evolution matrix."""
    H0 = det * _NOP + 0.5 * _ANH * (_NOP @ (_NOP - np.eye(3)))
    A_AP = _A + _ADAG
    A_AM = _A - _ADAG
    
    U = np.eye(3, dtype=complex)
    
    for i in range(_N):
        I = pulse_arr[i, 0]
        Q = pulse_arr[i, 1]
        H = H0 + 0.5 * scale * (I * A_AP + 1j * Q * A_AM)
        
        # Liouvillian generator A = -i H - Decay
        A_gen = -1j * H - _DECAY
        
        # Step evolution: U = exp(A_gen * dt) @ U
        U = expm(A_gen * _DT) @ U
        
    return U

def _fidelity(U, psi_init, psi_target):
    psi_final = U @ psi_init
    overlap = np.vdot(psi_target, psi_final)
    return np.abs(overlap)**2

def _score(pulse_arr):
    if not _valid(pulse_arr):
        return None
    scores = []
    for det in _DET_GRID:
        for scale in _SCALE_GRID:
            U = _propagate(pulse_arr, det, scale)
            fids = []
            for psi_i, psi_t in zip(_INIT_STATES, _TARGET_STATES):
                fids.append(_fidelity(U, psi_i, psi_t))
            scores.append(1.0 - np.mean(fids))
    return float(np.mean(scores) + np.max(scores))

def pulse(target_gate, hardware_model):
    if target_gate != 'X_pi':
        raise ValueError("Only X_pi supported")

    # baseline DRAG
    best_pulse = _make_pulse(0.5, 1.0)
    best_score = _score(best_pulse)
    if best_score is None:
        best_pulse = _make_pulse(0.0, 1.0)
        best_score = _score(best_pulse)
        if best_score is None:
             return np.zeros((_N, 2))

    # Coordinate descent
    alpha = 0.5
    amp = 1.0
    sigma = 40.0 / 6.0
    
    # Initial coarse search
    alphas = np.linspace(-0.5, 0.5, 10)
    amps = np.linspace(0.9, 1.1, 5)
    sigmas = np.array([3.0, 4.0, 5.0, 6.0, 7.0])
    
    for a in alphas:
        if _time_remaining() < 5.0: break
        for am in amps:
            if _time_remaining() < 5.0: break
            p = _make_pulse(a, am, sigma=sigma)
            s = _score(p)
            if s is not None and (best_score is None or s < best_score):
                best_score = s
                best_pulse = p
                alpha = a
                amp = am

    # Fine search with sigma
    for _ in range(20):
        if _time_remaining() < 10.0:
            break
        improved = False
        # alpha sweep
        for da in (-0.05, 0.05):
            a2 = alpha + da
            p = _make_pulse(a2, amp, sigma=sigma)
            s = _score(p)
            if s is not None and s < best_score:
                best_score = s
                best_pulse = p
                alpha = a2
                improved = True
        # amp sweep
        for da in (-0.02, 0.02):
            a2 = amp + da
            p = _make_pulse(alpha, a2, sigma=sigma)
            s = _score(p)
            if s is not None and s < best_score:
                best_score = s
                best_pulse = p
                amp = a2
                improved = True
        # sigma sweep
        for ds in (-0.5, 0.5):
            s2 = sigma + ds
            if s2 > 0:
                p = _make_pulse(alpha, amp, sigma=s2)
                sc = _score(p)
                if sc is not None and sc < best_score:
                    best_score = sc
                    best_pulse = p
                    sigma = s2
                    improved = True
        if not improved:
            break

    return best_pulse
# EVOLVE-BLOCK-END
