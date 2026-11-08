"""Rewritten program for robust X-pi gate control on a 3-level transmon."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.linalg import expm
from scipy.optimize import minimize

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

# Precompute A_AP and A_AM
_A_AP = _A + _ADAG
_A_AM = _A - _ADAG

# Precompute H0 for each detuning
_H0_DET = []
for det in _DET_GRID:
    H0 = det * _NOP + 0.5 * _ANH * (_NOP @ (_NOP - np.eye(3)))
    _H0_DET.append(H0)

def _make_pulse(alpha, amp, sigma_mult=1.0, center_mult=1.0):
    """Gaussian I-channel, DRAG Q = alpha * dI/dt. Area of I = pi."""
    t = np.arange(_N) * _DT
    sigma = (40.0 / 6.0) * sigma_mult
    center = (_T/2.0) * center_mult
    g = np.exp(-(t - center)**2 / (2.0 * sigma**2))
    area = np.sum(g) * _DT
    if area < 1e-12:
        area = 1e-12
    I = amp * g * (np.pi / area)
    dIdt = np.zeros(_N)
    dIdt[1:-1] = (I[2:] - I[:-2]) / (2.0 * _DT)
    dIdt[0] = (I[1] - I[0]) / _DT
    dIdt[-1] = (I[-1] - I[-2]) / _DT
    Q = alpha * dIdt
    return np.column_stack([I, Q])

def _valid(pulse_arr):
    r = np.sqrt(np.sum(pulse_arr**2, axis=1))
    if np.max(r) > _AMP + 1e-9:
        return False
    p0 = np.zeros(2)
    for i in range(_N):
        if np.linalg.norm(pulse_arr[i] - p0) / _DT > _SLEW + 1e-9:
            return False
        p0 = pulse_arr[i]
    if np.linalg.norm(p0) / _DT > _SLEW + 1e-9:
        return False
    return True

def _propagate(pulse_arr, det_idx, scale):
    """Return final 3x3 evolution matrix."""
    H0 = _H0_DET[det_idx]
    U = np.eye(3, dtype=complex)
    
    for i in range(_N):
        I = pulse_arr[i, 0]
        Q = pulse_arr[i, 1]
        H_drive = 0.5 * scale * (I * _A_AP + 1j * Q * _A_AM)
        H = H0 + H_drive
        
        A_gen = -1j * H - _DECAY
        U_step = expm(A_gen * _DT)
        U = U_step @ U
        
    return U

def _physics_score_fast(pulse_arr, det_indices=None, scales=None):
    """Calculate physics score, optionally over a subset of cases."""
    if det_indices is None:
        det_indices = range(len(_DET_GRID))
    if scales is None:
        scales = _SCALE_GRID
    
    scores = []
    for k in det_indices:
        for scale in scales:
            U = _propagate(pulse_arr, k, scale)
            fids = []
            for psi_i, psi_t in zip(_INIT_STATES, _TARGET_STATES):
                fids.append(_fidelity(U, psi_i, psi_t))
            s_case = 1.0 - np.mean(fids)
            scores.append(s_case)
    return float(np.mean(scores) + np.max(scores))

def _fidelity(U, psi_init, psi_target):
    psi_final = U @ psi_init
    overlap = np.vdot(psi_target, psi_final)
    return np.abs(overlap)**2

def _physics_score(pulse_arr):
    """Calculate the physics score (mean + max infidelity)."""
    return _physics_score_fast(pulse_arr)

def _penalty(pulse_arr):
    """Calculate penalty for constraint violations."""
    penalty = 0.0
    # Amplitude constraint
    r = np.sqrt(np.sum(pulse_arr**2, axis=1))
    amp_viol = np.maximum(0, r - _AMP)
    penalty += np.sum(amp_viol**2)
    
    # Slew constraint
    # From 0 to p[0]
    diff0 = np.linalg.norm(pulse_arr[0])
    slew_viol0 = np.maximum(0, diff0 / _DT - _SLEW)
    penalty += slew_viol0**2
    
    # Between steps
    diffs = np.diff(pulse_arr, axis=0)
    norms = np.sqrt(np.sum(diffs**2, axis=1))
    slew_viols = np.maximum(0, norms / _DT - _SLEW)
    penalty += np.sum(slew_viols**2)
    
    # From p[-1] to 0
    diffN = np.linalg.norm(pulse_arr[-1])
    slew_violN = np.maximum(0, diffN / _DT - _SLEW)
    penalty += slew_violN**2
    
    return penalty

def _total_loss(pulse_flat):
    """Loss function for optimizer. pulse_flat is 1D array of length 160."""
    pulse_arr = pulse_flat.reshape(_N, 2)
    phys = _physics_score(pulse_arr)
    pen = _penalty(pulse_arr)
    # Weight penalty heavily to ensure validity
    return phys + 10.0 * pen

def _make_pulse_params(alpha, amp, sigma_mult=1.0, center_mult=1.0, q_offset=0.0):
    """Create pulse with additional Q offset for robustness."""
    p = _make_pulse(alpha, amp, sigma_mult, center_mult)
    if q_offset != 0.0:
        p[:, 1] += q_offset
    return p

def _project(pulse_arr):
    """Project pulse to satisfy amplitude and slew constraints."""
    p = pulse_arr.copy()
    
    # 1. Amplitude clip
    for i in range(_N):
        r = np.linalg.norm(p[i])
        if r > _AMP:
            p[i] *= _AMP / r
            
    # 2. Slew constraint
    # Forward pass
    prev = np.zeros(2)
    for i in range(_N):
        diff = p[i] - prev
        norm = np.linalg.norm(diff)
        max_diff = _SLEW * _DT
        if norm > max_diff:
            p[i] = prev + diff * (max_diff / norm)
        prev = p[i]
        
    # Backward pass
    next_p = np.zeros(2)
    for i in range(_N - 1, -1, -1):
        diff = p[i] - next_p
        norm = np.linalg.norm(diff)
        max_diff = _SLEW * _DT
        if norm > max_diff:
            p[i] = next_p + diff * (max_diff / norm)
        next_p = p[i]
        
    # Final amplitude clip
    for i in range(_N):
        r = np.linalg.norm(p[i])
        if r > _AMP:
            p[i] *= _AMP / r
            
    return p

def _score_and_grad(pulse_flat):
    """Compute score and gradient using finite differences."""
    pulse_arr = pulse_flat.reshape(_N, 2)
    
    phys = _physics_score(pulse_arr)
    pen = _penalty(pulse_arr)
    loss = phys + 10.0 * pen
    
    # Numerical gradient
    grad = np.zeros_like(pulse_flat)
    eps = 1e-5
    for i in range(len(pulse_flat)):
        pulse_plus = pulse_flat.copy()
        pulse_plus[i] += eps
        loss_plus = _total_loss(pulse_plus)
        
        pulse_minus = pulse_flat.copy()
        pulse_minus[i] -= eps
        loss_minus = _total_loss(pulse_minus)
        
        grad[i] = (loss_plus - loss_minus) / (2 * eps)
    
    return loss, grad

def _loss(pulse_flat):
    """Loss function for optimizer."""
    return _total_loss(pulse_flat)

def pulse(target_gate, hardware_model):
    if target_gate != 'X_pi':
        raise ValueError("Only X_pi supported")
    
    # Try multiple initial DRAG configurations
    best_pulse = None
    best_score = float('inf')
    
    initial_configs = [
        (0.3, 1.0, 1.0, 1.0),
        (0.5, 1.0, 1.0, 1.0),
        (0.7, 1.0, 1.0, 1.0),
        (0.5, 1.0, 0.8, 1.0),
        (0.5, 1.0, 1.2, 1.0),
        (0.5, 0.9, 1.0, 1.0),
        (0.5, 1.1, 1.0, 1.0),
        (0.4, 1.0, 0.9, 1.0),
        (0.6, 1.0, 1.1, 1.0),
    ]
    
    for alpha, amp, sm, cm in initial_configs:
        if _time_remaining() < 120.0:
            break
        try:
            p = _make_pulse(alpha, amp, sm, cm)
            if _valid(p):
                s = _physics_score(p)
                if s < best_score:
                    best_score = s
                    best_pulse = p.copy()
        except Exception:
            pass
    
    if best_pulse is None:
        best_pulse = _make_pulse(0.5, 1.0)
        best_score = _physics_score(best_pulse)
    
    x0 = best_pulse.flatten()
    
    # Differential evolution for global search
    if _time_remaining() > 150.0:
        try:
            from scipy.optimize import differential_evolution
            bounds = [(-0.4, 0.4)] * 160
            # Use a subset of cases for speed during DE
            def _de_loss(x):
                p = x.reshape(_N, 2)
                pen = _penalty(p)
                if pen > 1e-6:
                    return phys_score + 100.0 * pen
                return _physics_score_fast(p, det_indices=[1], scales=[1.0]) + 10.0 * pen
            
            # DE is expensive, so use few iterations
            de_res = differential_evolution(_de_loss, bounds, 
                                            maxiter=50, popsize=10, 
                                            tol=1e-6, polish=False,
                                            seed=42)
            if de_res.fun < best_score:
                new_pulse = de_res.x.reshape(_N, 2)
                new_pulse = _project(new_pulse)
                new_score = _physics_score(new_pulse)
                if new_score < best_score:
                    best_pulse = new_pulse
                    best_score = new_score
                    x0 = best_pulse.flatten()
        except Exception:
            pass
    
    # Local refinement with L-BFGS-B
    if _time_remaining() > 60.0:
        try:
            res = minimize(_loss, x0, method='L-BFGS-B', 
                          options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8, 'disp': False})
            if res.fun < best_score:
                new_pulse = res.x.reshape(_N, 2)
                new_pulse = _project(new_pulse)
                new_score = _physics_score(new_pulse)
                if new_score < best_score:
                    best_pulse = new_pulse
                    best_score = new_score
                    x0 = best_pulse.flatten()
        except Exception:
            pass
    
    # Nelder-Mead refinement
    if _time_remaining() > 30.0:
        try:
            res = minimize(_loss, x0, method='Nelder-Mead',
                          options={'maxiter': 300, 'xatol': 1e-7, 'fatol': 1e-9, 'disp': False})
            if res.fun < best_score:
                new_pulse = res.x.reshape(_N, 2)
                new_pulse = _project(new_pulse)
                new_score = _physics_score(new_pulse)
                if new_score < best_score:
                    best_pulse = new_pulse
                    best_score = new_score
        except Exception:
            pass
    
    # Final fine-tuning
    if _time_remaining() > 15.0:
        try:
            res = minimize(_loss, x0, method='L-BFGS-B', 
                          options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10, 'disp': False})
            if res.fun < best_score:
                new_pulse = res.x.reshape(_N, 2)
                new_pulse = _project(new_pulse)
                new_score = _physics_score(new_pulse)
                if new_score < best_score:
                    best_pulse = new_pulse
                    best_score = new_score
        except Exception:
            pass
    
    # Ensure final result is valid
    if not _valid(best_pulse):
        best_pulse = _project(best_pulse)
        if not _valid(best_pulse):
            best_pulse = _make_pulse(0.5, 1.0)
            
    return best_pulse
# EVOLVE-BLOCK-END
