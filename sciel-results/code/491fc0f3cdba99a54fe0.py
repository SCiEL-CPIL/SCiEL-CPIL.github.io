"""Robust control pulses for a three-level transmon."""
import numpy as np
import time
import scipy.linalg as sla
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

# --- Constants ---
DT = 0.5
N_STEPS = 80
GATE_TIME = 40.0
T1 = 25000.0
Tphi = 40000.0
ANHARMONICITY = -2 * np.pi * 0.25
AMP_BOUND = 0.4
SLEW_BOUND = 0.12
DETUNING_VALUES = [-2 * np.pi * 0.002, 0.0, 2 * np.pi * 0.002]
SCALE_VALUES = [0.97, 1.0, 1.03]
DEADLINE = 315.0  # safety margin before 330s kill

# Precompute matrices
_a = np.array([[0, 1, 0],
               [0, 0, np.sqrt(2)],
               [0, 0, 0]], dtype=complex)
_adag = _a.conj().T
_n = _adag @ _a
_eye3 = np.eye(3, dtype=complex)

# Lindblad operators
_L1 = _a / np.sqrt(T1)
_L2 = _n * np.sqrt(2.0 / Tphi)

# Precompute Liouvillian building blocks
def _precompute_liouvillian_bases():
    I3 = _eye3
    L_det_base = -1j * (np.kron(_n, I3) - np.kron(I3, _n.T))
    H_I_base = 0.5 * (_a + _adag)
    L_I_base = -1j * (np.kron(H_I_base, I3) - np.kron(I3, H_I_base.T))
    H_Q_base = 0.5j * (_a - _adag)
    L_Q_base = -1j * (np.kron(H_Q_base, I3) - np.kron(I3, H_Q_base.T))
    H_anh = 0.5 * ANHARMONICITY * _n @ (_n - _eye3)
    L_anh_base = -1j * (np.kron(H_anh, I3) - np.kron(I3, H_anh.T))
    
    L_diss = np.zeros((9, 9), dtype=complex)
    for Lk in [_L1, _L2]:
        Lkd = Lk.conj().T
        LdL = Lkd @ Lk
        term1 = np.kron(Lkd.T, Lk)
        term2 = -0.5 * (np.kron(I3, LdL.T) + np.kron(LdL, I3))
        L_diss += term1 + term2
    
    return L_det_base, L_I_base, L_Q_base, L_anh_base, L_diss

_L_DET, _L_I, _L_Q, _L_ANH, _L_DISS = _precompute_liouvillian_bases()

# Initial states
_psi0 = np.array([1, 0, 0], dtype=complex)
_psi1 = np.array([0, 1, 0], dtype=complex)
_psi_plus = (_psi0 + _psi1) / np.sqrt(2)
_psi_minus = (_psi0 - _psi1) / np.sqrt(2)
_psi_plus_i = (_psi0 + 1j*_psi1) / np.sqrt(2)
_psi_minus_i = (_psi0 - 1j*_psi1) / np.sqrt(2)

# Ideal outputs (X applied to each)
_psi_outs = [
    np.array([0, 1, 0], dtype=complex),
    np.array([1, 0, 0], dtype=complex),
    (_psi0 + _psi1) / np.sqrt(2),
    (_psi0 - _psi1) / np.sqrt(2),
    (_psi1 + 1j*_psi0) / np.sqrt(2),
    (_psi1 - 1j*_psi0) / np.sqrt(2),
]

def _evolve_all_states(pulse_I, pulse_Q, detuning, scale):
    rhos = [np.outer(p, p.conj()) for p in [_psi0, _psi1, _psi_plus, _psi_minus, _psi_plus_i, _psi_minus_i]]
    rhos_vec = [r.flatten() for r in rhos]
    
    L_const = detuning * _L_DET + _L_ANH + _L_DISS
    
    for i in range(N_STEPS):
        I_val = pulse_I[i]
        Q_val = pulse_Q[i]
        
        L = L_const + scale * I_val * _L_I + scale * Q_val * _L_Q
        expL = sla.expm(L * DT)
        
        for j in range(6):
            rhos_vec[j] = expL @ rhos_vec[j]
    
    return [v.reshape(3, 3) for v in rhos_vec]

def _fidelity_from_rhos(rhos):
    fidelities = []
    for rho, psi_out in zip(rhos, _psi_outs):
        F = np.real(np.vdot(psi_out, rho @ psi_out))
        F = max(0.0, min(1.0, F))
        fidelities.append(F)
    return np.mean(fidelities)

def _compute_score(pulse_I, pulse_Q):
    infids = []
    for det in DETUNING_VALUES:
        for scale in SCALE_VALUES:
            rhos = _evolve_all_states(pulse_I, pulse_Q, det, scale)
            F = _fidelity_from_rhos(rhos)
            infids.append(1.0 - F)
    return np.mean(infids) + np.max(infids)

def _is_valid(pulse_I, pulse_Q):
    amp = np.sqrt(pulse_I**2 + pulse_Q**2)
    if np.any(amp > AMP_BOUND + 1e-12):
        return False
    for i in range(N_STEPS - 1):
        dI = pulse_I[i+1] - pulse_I[i]
        dQ = pulse_Q[i+1] - pulse_Q[i]
        slew = np.sqrt(dI**2 + dQ**2) / DT
        if slew > SLEW_BOUND + 1e-12:
            return False
    slew0 = np.sqrt(pulse_I[0]**2 + pulse_Q[0]**2) / DT
    if slew0 > SLEW_BOUND + 1e-12:
        return False
    slewl = np.sqrt(pulse_I[-1]**2 + pulse_Q[-1]**2) / DT
    if slewl > SLEW_BOUND + 1e-12:
        return False
    return True

def _enforce_constraints(pulse_I, pulse_Q):
    pulse_I = pulse_I.copy()
    pulse_Q = pulse_Q.copy()
    amp = np.sqrt(pulse_I**2 + pulse_Q**2)
    mask = amp > AMP_BOUND
    if np.any(mask):
        pulse_I[mask] *= AMP_BOUND / amp[mask]
        pulse_Q[mask] *= AMP_BOUND / amp[mask]
    for i in range(N_STEPS - 1):
        dI = pulse_I[i+1] - pulse_I[i]
        dQ = pulse_Q[i+1] - pulse_Q[i]
        slew = np.sqrt(dI**2 + dQ**2) / DT
        if slew > SLEW_BOUND:
            scale = SLEW_BOUND * DT / slew
            pulse_I[i+1] = pulse_I[i] + dI * scale
            pulse_Q[i+1] = pulse_Q[i] + dQ * scale
    slew0 = np.sqrt(pulse_I[0]**2 + pulse_Q[0]**2) / DT
    if slew0 > SLEW_BOUND:
        scale = SLEW_BOUND * DT / slew0
        pulse_I[0] *= scale
        pulse_Q[0] *= scale
    slewl = np.sqrt(pulse_I[-1]**2 + pulse_Q[-1]**2) / DT
    if slewl > SLEW_BOUND:
        scale = SLEW_BOUND * DT / slewl
        pulse_I[-1] *= scale
        pulse_Q[-1] *= scale
    return pulse_I, pulse_Q

def _seed_pulse():
    """Generate DRAG-style Gaussian seed with analytically optimal beta."""
    t = np.arange(N_STEPS) * DT
    sigma = GATE_TIME / 6.0
    center = GATE_TIME / 2.0
    
    g = np.exp(-0.5 * (t - center)**2 / sigma**2)
    g[0] = 0.0
    g[-1] = 0.0
    
    area = np.sum(g) * DT
    if area > 0:
        g *= np.pi / area
    
    # Analytically motivated beta: beta = omega_01 * sigma^2 / (2*sqrt(2)) * (alpha/omega_01)
    # For DRAG: beta = alpha * sigma^2 / (2 * sqrt(2)) where alpha is anharmonicity
    # Actually the standard formula is beta = -alpha * sigma^2 / (2 * sqrt(2))
    # alpha = -2*pi*0.25, so beta = 2*pi*0.25 * sigma^2 / (2*sqrt(2))
    beta = (-ANHARMONICITY) * sigma**2 / (2.0 * np.sqrt(2.0))
    
    dI_dt = np.gradient(g, DT)
    
    pulse_I = g.copy()
    pulse_Q = beta * dI_dt
    
    pulse_I, pulse_Q = _enforce_constraints(pulse_I, pulse_Q)
    return pulse_I, pulse_Q

def _pulse_from_params(params):
    """
    Generate pulse from parameterization with more degrees of freedom.
    params: [sigma, beta, amp_scale, center_shift, q_offset, q_width, q_center]
    """
    sigma = params[0]
    beta = params[1]
    amp_scale = params[2]
    center_shift = params[3]
    q_offset = params[4]
    q_width = params[5]
    q_center = params[6]
    
    t = np.arange(N_STEPS) * DT
    center = GATE_TIME / 2.0 + center_shift
    
    g = np.exp(-0.5 * (t - center)**2 / sigma**2)
    g[0] = 0.0
    g[-1] = 0.0
    
    area = np.sum(g) * DT
    if area > 0:
        g *= np.pi * amp_scale / area
    
    dI_dt = np.gradient(g, DT)
    
    pulse_I = g.copy()
    pulse_Q = beta * dI_dt
    
    # Add a separate Gaussian component to Q for more flexibility
    if q_width > 0.1:
        gq = np.exp(-0.5 * (t - q_center)**2 / q_width**2)
        gq[0] = 0.0
        gq[-1] = 0.0
        pulse_Q += q_offset * gq
    
    pulse_I, pulse_Q = _enforce_constraints(pulse_I, pulse_Q)
    return pulse_I, pulse_Q

def _score_from_params(params):
    pulse_I, pulse_Q = _pulse_from_params(params)
    if not _is_valid(pulse_I, pulse_Q):
        return 10.0
    return _compute_score(pulse_I, pulse_Q)

def _nelder_mead_optimize(pulse_I, pulse_Q, time_limit=100):
    """Optimize using Nelder-Mead on low-dimensional parameter space."""
    start_time = time.time()
    
    best_I = pulse_I.copy()
    best_Q = pulse_Q.copy()
    best_score = _compute_score(best_I, best_Q)
    
    # Parameter bounds for Nelder-Mead
    bounds = [
        (3.0, 25.0),    # sigma
        (-1.0, 1.0),    # beta
        (0.5, 1.5),     # amp_scale
        (-5.0, 5.0),    # center_shift
        (-0.3, 0.3),    # q_offset
        (2.0, 20.0),    # q_width
        (5.0, 35.0),    # q_center
    ]
    
    # Multiple starting points
    np.random.seed(42)
    n_starts = 5
    best_overall = best_score
    
    for trial in range(n_starts):
        if time.time() - start_time > time_limit:
            break
        
        # Generate starting point
        if trial == 0:
            # Start from seed parameters
            sigma0 = GATE_TIME / 6.0
            beta0 = (-ANHARMONICITY) * sigma0**2 / (2.0 * np.sqrt(2.0))
            x0 = np.array([sigma0, beta0, 1.0, 0.0, 0.0, 10.0, 20.0])
        else:
            # Random starting points
            x0 = np.array([
                np.random.uniform(*bounds[0]),
                np.random.uniform(*bounds[1]),
                np.random.uniform(*bounds[2]),
                np.random.uniform(*bounds[3]),
                np.random.uniform(*bounds[4]),
                np.random.uniform(*bounds[5]),
                np.random.uniform(*bounds[6]),
            ])
        
        # Clip to bounds
        x0 = np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds])
        
        def objective(params):
            if time.time() - start_time > time_limit:
                return 100.0
            return _score_from_params(params)
        
        try:
            result = minimize(objective, x0, method='Nelder-Mead',
                            options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8,
                                    'adaptive': True})
            
            if result.fun < best_overall:
                best_overall = result.fun
                pulse_I, pulse_Q = _pulse_from_params(result.x)
                if _is_valid(pulse_I, pulse_Q):
                    best_I = pulse_I.copy()
                    best_Q = pulse_Q.copy()
                    best_score = result.fun
        except Exception:
            pass
    
    return best_I, best_Q, best_score

def _dop853_recheck(pulse_I, pulse_Q):
    """Recheck nominal case with DOP853, return max difference."""
    det = 0.0
    scale = 1.0
    
    rhos_ours = _evolve_all_states(pulse_I, pulse_Q, det, scale)
    
    def liouvillian_vec(t, rho_vec):
        rho = rho_vec.reshape(3, 3)
        idx = int(t / DT)
        if idx >= N_STEPS:
            idx = N_STEPS - 1
        I_val = pulse_I[idx]
        Q_val = pulse_Q[idx]
        
        L = (det * _L_DET + _L_ANH + _L_DISS 
             + scale * I_val * _L_I + scale * Q_val * _L_Q)
        return L @ rho_vec
    
    max_diff = 0.0
    for idx, psi in enumerate([_psi0, _psi1, _psi_plus, _psi_minus, _psi_plus_i, _psi_minus_i]):
        rho_init = np.outer(psi, psi.conj()).flatten()
        
        sol = solve_ivp(liouvillian_vec, [0, GATE_TIME], rho_init, 
                       method='DOP853', rtol=1e-12, atol=1e-14, max_step=0.1)
        
        rho_dop = sol.y[:, -1].reshape(3, 3)
        diff = np.max(np.abs(rho_dop - rhos_ours[idx]))
        max_diff = max(max_diff, diff)
    
    return max_diff

def pulse(target_gate, hardware_model):
    """Return an (80, 2) array of I and Q amplitudes for an X-pi gate."""
    if target_gate != 'X_pi':
        return np.zeros((80, 2))
    
    start_time = time.time()
    
    # Generate seed pulse with DRAG
    pulse_I, pulse_Q = _seed_pulse()
    seed_score = _compute_score(pulse_I, pulse_Q)
    
    # Phase 1: Nelder-Mead optimization on low-dimensional parameters
    elapsed = time.time() - start_time
    nm_time_limit = min(180, DEADLINE - elapsed - 30)
    if nm_time_limit > 30:
        pulse_I, pulse_Q, nm_score = _nelder_mead_optimize(pulse_I, pulse_Q, time_limit=nm_time_limit)
    else:
        nm_score = seed_score
    
    # DOP853 recheck
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 15.0:
        diff = _dop853_recheck(pulse_I, pulse_Q)
        if diff > 2e-7:
            # If recheck fails, try one more optimization
            remaining = DEADLINE - (time.time() - start_time) - 5.0
            if remaining > 15:
                pulse_I, pulse_Q, _ = _nelder_mead_optimize(pulse_I, pulse_Q, time_limit=remaining)
    
    result = np.column_stack((pulse_I, pulse_Q))
    
    return result
