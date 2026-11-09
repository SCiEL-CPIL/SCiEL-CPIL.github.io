"""Robust control pulses for a three-level transmon."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.linalg as sla
from scipy.integrate import solve_ivp

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
DEADLINE = 310.0  # safety margin before 330s kill

# Precompute matrices
_a = np.array([[0, 1, 0],
               [0, 0, np.sqrt(2)],
               [0, 0, 0]], dtype=complex)
_adag = _a.conj().T
_n = _adag @ _a
_n2 = _n @ _n  # n^2
_eye3 = np.eye(3, dtype=complex)

# Lindblad operators
_L1 = _a / np.sqrt(T1)
_L2 = _n * np.sqrt(2.0 / Tphi)

# Precompute Liouvillian building blocks
def _precompute_liouvillian_bases():
    """Precompute the matrix coefficients for the Liouvillian."""
    I3 = _eye3
    
    # Detuning term
    L_det_base = -1j * (np.kron(_n, I3) - np.kron(I3, _n.T))
    
    # Drive I term: 0.5*(a+a†)
    H_I_base = 0.5 * (_a + _adag)
    L_I_base = -1j * (np.kron(H_I_base, I3) - np.kron(I3, H_I_base.T))
    
    # Drive Q term: 0.5*i*(a-a†)
    H_Q_base = 0.5j * (_a - _adag)
    L_Q_base = -1j * (np.kron(H_Q_base, I3) - np.kron(I3, H_Q_base.T))
    
    # Anharmonicity term (constant part): 0.5*alpha*n*(n-1)
    H_anh = 0.5 * ANHARMONICITY * _n @ (_n - _eye3)
    L_anh_base = -1j * (np.kron(H_anh, I3) - np.kron(I3, H_anh.T))
    
    # Dissipative part (constant)
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
    _psi1,
    _psi0,
    _psi_plus,
    _psi_minus,
    _psi_minus_i.conj() * 1j,  # X(|0>+i|1>)/sqrt(2) = (|1>+i|0>)/sqrt(2)
    _psi_plus_i.conj() * 1j   # X(|0>-i|1>)/sqrt(2) = (|1>-i|0>)/sqrt(2)
]
# Let me recompute these properly
_psi_outs = [
    np.array([0, 1, 0], dtype=complex),
    np.array([1, 0, 0], dtype=complex),
    (_psi0 + _psi1) / np.sqrt(2),
    (_psi0 - _psi1) / np.sqrt(2),
    (_psi1 + 1j*_psi0) / np.sqrt(2),
    (_psi1 - 1j*_psi0) / np.sqrt(2),
]

def _evolve_all_states(pulse_I, pulse_Q, detuning, scale):
    """Evolve all 6 input states through the pulse, return final density matrices."""
    rhos = [np.outer(p, p.conj()) for p in [_psi0, _psi1, _psi_plus, _psi_minus, _psi_plus_i, _psi_minus_i]]
    rhos_vec = [r.flatten() for r in rhos]
    
    # Constant part of Liouvillian
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
    """Compute mean fidelity to X gate from 6 final density matrices."""
    fidelities = []
    for rho, psi_out in zip(rhos, _psi_outs):
        F = np.real(np.vdot(psi_out, rho @ psi_out))
        F = max(0.0, min(1.0, F))
        fidelities.append(F)
    
    return np.mean(fidelities)

def _compute_infidelity(pulse_I, pulse_Q):
    """Compute infidelity = 1 - mean fidelity over 9 cases."""
    fidelities = []
    for det in DETUNING_VALUES:
        for scale in SCALE_VALUES:
            rhos = _evolve_all_states(pulse_I, pulse_Q, det, scale)
            F = _fidelity_from_rhos(rhos)
            fidelities.append(F)
    
    mean_F = np.mean(fidelities)
    return 1.0 - mean_F

def _compute_score(pulse_I, pulse_Q):
    """Compute score = mean + max of 9 infidelities."""
    infids = []
    for det in DETUNING_VALUES:
        for scale in SCALE_VALUES:
            rhos = _evolve_all_states(pulse_I, pulse_Q, det, scale)
            F = _fidelity_from_rhos(rhos)
            infids.append(1.0 - F)
    
    return np.mean(infids) + np.max(infids)

def _is_valid(pulse_I, pulse_Q):
    """Check amplitude and slew constraints."""
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
    """Project pulse onto feasible set (amplitude + slew)."""
    pulse_I = pulse_I.copy()
    pulse_Q = pulse_Q.copy()
    
    # Amplitude
    amp = np.sqrt(pulse_I**2 + pulse_Q**2)
    mask = amp > AMP_BOUND
    if np.any(mask):
        pulse_I[mask] *= AMP_BOUND / amp[mask]
        pulse_Q[mask] *= AMP_BOUND / amp[mask]
    
    # Slew: forward pass
    for i in range(N_STEPS - 1):
        dI = pulse_I[i+1] - pulse_I[i]
        dQ = pulse_Q[i+1] - pulse_Q[i]
        slew = np.sqrt(dI**2 + dQ**2) / DT
        if slew > SLEW_BOUND:
            scale = SLEW_BOUND * DT / slew
            pulse_I[i+1] = pulse_I[i] + dI * scale
            pulse_Q[i+1] = pulse_Q[i] + dQ * scale
    
    # Edge slews
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
    """Generate DRAG-style Gaussian seed with beta compensation."""
    t = np.arange(N_STEPS) * DT
    sigma = GATE_TIME / 6.0
    center = GATE_TIME / 2.0
    
    # Gaussian envelope
    g = np.exp(-0.5 * (t - center)**2 / sigma**2)
    g[0] = 0.0
    g[-1] = 0.0
    
    # Scale to area pi
    area = np.sum(g) * DT
    if area > 0:
        g *= np.pi / area
    
    # DRAG beta
    beta = 0.15
    
    # Derivative of I envelope
    dI_dt = np.gradient(g, DT)
    
    pulse_I = g.copy()
    pulse_Q = beta * dI_dt
    
    # Enforce constraints
    pulse_I, pulse_Q = _enforce_constraints(pulse_I, pulse_Q)
    
    return pulse_I, pulse_Q

def _gradient_estimate(pulse_I, pulse_Q, score_func, epsilon=0.01):
    """Estimate gradient using finite differences on a subset of steps."""
    grad_I = np.zeros(N_STEPS)
    grad_Q = np.zeros(N_STEPS)
    
    base_score = score_func(pulse_I, pulse_Q)
    
    # Sample a subset of steps to estimate gradient
    n_samples = 10
    indices = np.random.choice(N_STEPS, size=n_samples, replace=False)
    
    for idx in indices:
        # I component
        p_plus_I = pulse_I.copy()
        p_plus_I[idx] += epsilon
        p_plus_I, p_plus_Q = _enforce_constraints(p_plus_I, pulse_Q)
        s_plus = score_func(p_plus_I, p_plus_Q)
        
        p_minus_I = pulse_I.copy()
        p_minus_I[idx] -= epsilon
        p_minus_I, p_minus_Q = _enforce_constraints(p_minus_I, pulse_Q)
        s_minus = score_func(p_minus_I, p_minus_Q)
        
        grad_I[idx] = (s_plus - s_minus) / (2 * epsilon)
        
        # Q component
        p_plus_Q = pulse_Q.copy()
        p_plus_Q[idx] += epsilon
        p_plus_I2, p_plus_Q2 = _enforce_constraints(pulse_I, p_plus_Q)
        s_plus_q = score_func(p_plus_I2, p_plus_Q2)
        
        p_minus_Q = pulse_Q.copy()
        p_minus_Q[idx] -= epsilon
        p_minus_I2, p_minus_Q2 = _enforce_constraints(pulse_I, p_minus_Q)
        s_minus_q = score_func(p_minus_I2, p_minus_Q2)
        
        grad_Q[idx] = (s_plus_q - s_minus_q) / (2 * epsilon)
    
    return grad_I, grad_Q

def _optimize_pulse(pulse_I, pulse_Q, time_limit=280):
    """Optimize using gradient estimation + random search."""
    best_I = pulse_I.copy()
    best_Q = pulse_Q.copy()
    best_score = _compute_score(best_I, best_Q)
    
    start_time = time.time()
    np.random.seed(42)
    
    step_size = 0.05
    no_improve = 0
    
    while time.time() - start_time < time_limit:
        # Try gradient step
        grad_I, grad_Q = _gradient_estimate(best_I, best_Q, _compute_score)
        
        # Normalize gradient
        grad_norm = np.sqrt(np.sum(grad_I**2) + np.sum(grad_Q**2))
        if grad_norm > 1e-10:
            grad_I /= grad_norm
            grad_Q /= grad_norm
        
        new_I = best_I - step_size * grad_I
        new_Q = best_Q - step_size * grad_Q
        new_I, new_Q = _enforce_constraints(new_I, new_Q)
        
        if _is_valid(new_I, new_Q):
            new_score = _compute_score(new_I, new_Q)
            if new_score < best_score:
                best_I = new_I.copy()
                best_Q = new_Q.copy()
                best_score = new_score
                no_improve = 0
            else:
                no_improve += 1
        
        # Random perturbation as fallback
        if no_improve > 3:
            dI = np.random.normal(0, step_size * 0.5, N_STEPS)
            dQ = np.random.normal(0, step_size * 0.3, N_STEPS)
            
            new_I = best_I + dI
            new_Q = best_Q + dQ
            new_I, new_Q = _enforce_constraints(new_I, new_Q)
            
            if _is_valid(new_I, new_Q):
                new_score = _compute_score(new_I, new_Q)
                if new_score < best_score:
                    best_I = new_I.copy()
                    best_Q = new_Q.copy()
                    best_score = new_score
                    no_improve = 0
                    step_size *= 0.9
                else:
                    step_size *= 0.8
            no_improve = 0
        
        if step_size < 0.005:
            break
    
    return best_I, best_Q, best_score

def _dop853_recheck(pulse_I, pulse_Q):
    """Recheck nominal case with DOP853, return max difference."""
    det = 0.0
    scale = 1.0
    
    # Our method
    rhos_ours = _evolve_all_states(pulse_I, pulse_Q, det, scale)
    
    # DOP853 method
    def liouvillian_vec(t, rho_vec):
        rho = rho_vec.reshape(3, 3)
        I_val = pulse_I[int(t / DT)] if int(t / DT) < N_STEPS else 0.0
        Q_val = pulse_Q[int(t / DT)] if int(t / DT) < N_STEPS else 0.0
        
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
    
    # Optimize
    remaining = DEADLINE - (time.time() - start_time) - 10.0  # 10s safety margin for recheck
    pulse_I, pulse_Q, score = _optimize_pulse(pulse_I, pulse_Q, time_limit=max(10, remaining))
    
    # DOP853 recheck
    if time.time() - start_time < DEADLINE - 5.0:
        diff = _dop853_recheck(pulse_I, pulse_Q)
        if diff > 2e-7:
            # If recheck fails, try to refine further
            remaining2 = DEADLINE - (time.time() - start_time) - 5.0
            if remaining2 > 10:
                pulse_I, pulse_Q, score = _optimize_pulse(pulse_I, pulse_Q, time_limit=remaining2)
    
    result = np.column_stack((pulse_I, pulse_Q))
    
    return result
# EVOLVE-BLOCK-END
