"""Robust control pulses for a three-level transmon."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.linalg as sla

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
DEADLINE = 320.0  # safety margin before 330s kill

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
# L(rho) = -i[H, rho] + diss
# H = detuning*n + 0.5*anh*n*(n-1) + 0.5*scale*(I*(a+a†) + iQ*(a-a†))
# Coherent part is linear in (detuning, scale*I, scale*Q)
# So L = L_det * detuning + L_I * (scale*I) + L_Q * (scale*Q) + L_diss_const

def _precompute_liouvillian_bases():
    """Precompute the matrix coefficients for the Liouvillian."""
    I3 = _eye3
    
    # Detuning term: -i(detuning*n, rho) = -i(detuning)(n⊗I - I⊗n^T)
    L_det_base = -1j * (np.kron(_n, I3) - np.kron(I3, _n.T))
    
    # Drive I term: 0.5*(a+a†)
    H_I_base = 0.5 * (_a + _adag)
    L_I_base = -1j * (np.kron(H_I_base, I3) - np.kron(I3, H_I_base.T))
    
    # Drive Q term: 0.5*i*(a-a†)
    H_Q_base = 0.5j * (_a - _adag)
    L_Q_base = -1j * (np.kron(H_Q_base, I3) - np.kron(I3, H_Q_base.T))
    
    # Dissipative part (constant)
    L_diss = np.zeros((9, 9), dtype=complex)
    for Lk in [_L1, _L2]:
        Lkd = Lk.conj().T
        LdL = Lkd @ Lk
        term1 = np.kron(Lkd.T, Lk)
        term2 = -0.5 * (np.kron(I3, LdL.T) + np.kron(LdL, I3))
        L_diss += term1 + term2
    
    return L_det_base, L_I_base, L_Q_base, L_diss

_L_DET, _L_I, _L_Q, _L_DISS = _precompute_liouvillian_bases()

def _evolve_all_states(pulse_I, pulse_Q, detuning, scale):
    """Evolve all 6 input states through the pulse, return final density matrices.
    Uses the precomputed Liouvillian bases for speed.
    Returns list of 6 final rho matrices."""
    
    # Build the 9x9 Liouvillian for this (detuning, scale)
    # L = detuning*L_DET + scale*I*L_I + scale*Q*L_Q + L_DISS
    # But I and Q vary per step, so we build per step.
    
    # Initial states
    psi0 = np.array([1, 0, 0], dtype=complex)
    psi1 = np.array([0, 1, 0], dtype=complex)
    psi_plus = (psi0 + psi1) / np.sqrt(2)
    psi_minus = (psi0 - psi1) / np.sqrt(2)
    psi_plus_i = (psi0 + 1j*psi1) / np.sqrt(2)
    psi_minus_i = (psi0 - 1j*psi1) / np.sqrt(2)
    
    # Start with rho = |psi><psi| for each
    rhos = [np.outer(p, p.conj()) for p in [psi0, psi1, psi_plus, psi_minus, psi_plus_i, psi_minus_i]]
    rhos_vec = [r.flatten() for r in rhos]
    
    for i in range(N_STEPS):
        I_val = pulse_I[i]
        Q_val = pulse_Q[i]
        
        # Build Liouvillian: L = detuning*L_DET + scale*I_val*L_I + scale*Q_val*L_Q + L_DISS
        L = (detuning * _L_DET 
             + scale * I_val * _L_I 
             + scale * Q_val * _L_Q 
             + _L_DISS)
        
        expL = sla.expm(L * DT)
        
        for j in range(6):
            rhos_vec[j] = expL @ rhos_vec[j]
    
    return [v.reshape(3, 3) for v in rhos_vec]

def _fidelity_from_rhos(rhos):
    """Compute mean fidelity to X gate from 6 final density matrices."""
    X = np.array([[0, 1, 0],
                  [1, 0, 0],
                  [0, 0, 0]], dtype=complex)
    
    # Ideal outputs: X|psi> for each input state
    # |0> -> |1>, |1> -> |0>, (|0>+|1>)/√2 -> (|1>+|0>)/√2, etc.
    psi_outs = [
        np.array([0, 1, 0], dtype=complex),   # X|0>
        np.array([1, 0, 0], dtype=complex),   # X|1>
        (np.array([1, 0, 0], dtype=complex) + np.array([0, 1, 0], dtype=complex)) / np.sqrt(2),  # X(|0>+|1>)/√2
        (np.array([1, 0, 0], dtype=complex) - np.array([0, 1, 0], dtype=complex)) / np.sqrt(2),  # X(|0>-|1>)/√2
        (np.array([0, 1, 0], dtype=complex) + 1j * np.array([1, 0, 0], dtype=complex)) / np.sqrt(2),  # X(|0>+i|1>)/√2
        (np.array([0, 1, 0], dtype=complex) - 1j * np.array([1, 0, 0], dtype=complex)) / np.sqrt(2),  # X(|0>-i|1>)/√2
    ]
    
    fidelities = []
    for rho, psi_out in zip(rhos, psi_outs):
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
    
    # DRAG beta: beta = -delta_01 / (2 * pi * delta_12) approximately
    # For our case, the anharmonicity causes leakage to |2>
    # beta ≈ -alpha / (2 * drive_freq) but we use a heuristic
    # Standard DRAG: Q component = beta * dI/dt
    # For a transmon with anharmonicity alpha, beta ≈ alpha/(2*omega_01) * something
    # Let's use a reasonable beta that helps cancel |2> leakage
    # alpha = -2*pi*0.25, so |alpha| = pi/2
    # A common choice: beta = -alpha / (2 * pi * f) where f is related to drive
    # Let's try beta = 0.5 * |alpha| / (2*pi) = 0.5 * 0.25 / 2 = 0.0625
    # Actually, for a pi pulse, beta ≈ alpha * tau / (4*pi) where tau is pulse duration
    # Let's just use a tunable beta
    beta = 0.15  # Will be optimized
    
    # Derivative of I envelope
    dI_dt = np.gradient(g, DT)
    
    pulse_I = g.copy()
    pulse_Q = beta * dI_dt
    
    # Enforce constraints
    pulse_I, pulse_Q = _enforce_constraints(pulse_I, pulse_Q)
    
    return pulse_I, pulse_Q

def _optimize_pulse(pulse_I, pulse_Q, max_iters=150, time_limit=280):
    """Optimize using random restarts + local refinement."""
    best_I = pulse_I.copy()
    best_Q = pulse_Q.copy()
    best_infid = _compute_infidelity(best_I, best_Q)
    
    start_time = time.time()
    np.random.seed(42)
    
    # Phase 1: Random search with adaptive step size
    step_size = 0.08
    no_improve = 0
    
    for it in range(max_iters):
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            break
        
        # Adapt step size
        if no_improve > 10:
            step_size *= 0.8
            no_improve = 0
            if step_size < 0.01:
                step_size = 0.01
        
        dI = np.random.normal(0, step_size, N_STEPS)
        dQ = np.random.normal(0, step_size * 0.5, N_STEPS)
        
        new_I = best_I + dI
        new_Q = best_Q + dQ
        new_I, new_Q = _enforce_constraints(new_I, new_Q)
        
        if not _is_valid(new_I, new_Q):
            continue
        
        new_infid = _compute_infidelity(new_I, new_Q)
        
        if new_infid < best_infid:
            best_I = new_I.copy()
            best_Q = new_Q.copy()
            best_infid = new_infid
            no_improve = 0
        else:
            no_improve += 1
    
    return best_I, best_Q, best_infid

def pulse(target_gate, hardware_model):
    """Return an (80, 2) array of I and Q amplitudes for an X-pi gate."""
    if target_gate != 'X_pi':
        return np.zeros((80, 2))
    
    start_time = time.time()
    
    # Generate seed pulse with DRAG
    pulse_I, pulse_Q = _seed_pulse()
    
    # Optimize
    remaining = DEADLINE - (time.time() - start_time) - 5.0  # 5s safety margin
    pulse_I, pulse_Q, infid = _optimize_pulse(pulse_I, pulse_Q, max_iters=150, time_limit=max(10, remaining))
    
    result = np.column_stack((pulse_I, pulse_Q))
    
    return result
# EVOLVE-BLOCK-END
