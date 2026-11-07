"""Optimized pulse for three-level transmon X_pi gate with robustness."""
import numpy as np
import time
from scipy.linalg import expm
from scipy.optimize import minimize

_DEADLINE = time.time() + 305  # safety margin before 330s kill

# Hardware parameters
T1 = 25000.0
Tphi = 40000.0
ANHARMONICITY = -2 * np.pi * 0.25
AMPLITUDE_BOUND = 0.4
SLEW_BOUND = 0.12
N_STEPS = 80
DT = 0.5

# Precompute ladder operators
_a = np.zeros((3, 3), dtype=complex)
_a[1, 0] = np.sqrt(1.0)
_a[2, 1] = np.sqrt(2.0)
_a_dag = _a.conj().T
_n_op = _a_dag @ _a

# Lindblad operators
_L1 = _a / np.sqrt(T1)
_L2 = _n_op * np.sqrt(2.0 / Tphi)
_L1_dag = _L1.conj().T
_L2_dag = _L2.conj().T

# Precompute dissipation terms
_L1_dag_L1 = _L1_dag @ _L1
_L2_dag_L2 = _L2_dag @ _L2

_I3 = np.eye(3)

# Precompute dissipation superoperator (constant part)
_diss_super = np.zeros((9, 9), dtype=complex)
for Lk, Lk_dag, Lk_dag_Lk in [(_L1, _L1_dag, _L1_dag_L1), (_L2, _L2_dag, _L2_dag_L2)]:
    _diss_super += np.kron(Lk_dag.T, Lk) - 0.5 * (np.kron(_I3, Lk_dag_Lk.T) + np.kron(Lk_dag_Lk, _I3))

# Precompute Hamiltonian basis matrices for I and Q
# H_drive_I = 0.5 * (a + a^dag)
_H_I = 0.5 * (_a + _a_dag)
# H_drive_Q = 0.5 * 1j * (a - a^dag)
_H_Q = 0.5 * 1j * (_a - _a_dag)

# Precompute superoperator contributions for I and Q (without scale)
# L_ham_I = -1j * (kron(I3, H_I.T) - kron(H_I.T, I3))
_L_ham_I = -1j * (np.kron(_I3, _H_I.T) - np.kron(_H_I.T, _I3))
_L_ham_Q = -1j * (np.kron(_I3, _H_Q.T) - np.kron(_H_Q.T, _I3))

# Precompute detuning superoperator
_L_det = -1j * (np.kron(_I3, _n_op.T) - np.kron(_n_op.T, _I3))

# Precompute anharmonicity superoperator
_H_anh = 0.5 * ANHARMONICITY * (_n_op @ (_n_op - _I3))
_L_anh = -1j * (np.kron(_I3, _H_anh.T) - np.kron(_H_anh.T, _I3))

# Precompute static Liouvillian (detuning + anharmonicity + dissipation) for each (det, scale) pair
# Actually, detuning and anharmonicity are static, only I and Q vary with scale
# L_total = L_det + L_anh + L_diss + scale * (I * L_ham_I + Q * L_ham_Q)

# Precompute for each detuning value
_detunings_list = [-2*np.pi*0.002, 0.0, 2*np.pi*0.002]
_scales_list = [0.97, 1.0, 1.03]

# Precompute static Liouvillian for each detuning
_static_liouvillian = {}
for det in _detunings_list:
    L_static = det * _L_det + _L_anh + _diss_super
    _static_liouvillian[det] = L_static


def _build_liouvillian_fast(detuning, scale, I, Q):
    """Build the Liouvillian superoperator using precomputed components."""
    L = _static_liouvillian[detuning].copy()
    L += scale * (I * _L_ham_I + Q * _L_ham_Q)
    return L


def _simulate_pulse(pulse, detuning, scale):
    """Simulate the pulse and return the final density matrix."""
    rho = np.zeros((3, 3), dtype=complex)
    rho[0, 0] = 1.0
    
    for i in range(N_STEPS):
        I_val = pulse[i, 0]
        Q_val = pulse[i, 1]
        L = _build_liouvillian_fast(detuning, scale, I_val, Q_val)
        L_mat = expm(L * DT)
        rho_vec = rho.flatten()
        rho_vec = L_mat @ rho_vec
        rho = rho_vec.reshape(3, 3)
    
    return rho


def _fidelity(rho, target_state):
    """Compute fidelity between density matrix and pure target state."""
    val = np.vdot(target_state, rho @ target_state).real
    return np.clip(val, 0.0, 1.0)


def _check_validity(pulse):
    """Check amplitude and slew rate constraints."""
    for i in range(N_STEPS):
        amp = np.sqrt(pulse[i, 0]**2 + pulse[i, 1]**2)
        if amp > AMPLITUDE_BOUND + 1e-9:
            return False
    
    prev = np.array([0.0, 0.0])
    for i in range(N_STEPS):
        curr = pulse[i]
        if np.linalg.norm(curr - prev) / DT > SLEW_BOUND + 1e-9:
            return False
        prev = curr
    if np.linalg.norm(prev) / DT > SLEW_BOUND + 1e-9:
        return False
    
    return True


def _compute_score(pulse):
    """Compute the full score for a pulse."""
    if not _check_validity(pulse):
        return None
    
    target_states = [
        np.array([1, 0, 0], dtype=complex),
        np.array([0, 1, 0], dtype=complex),
        np.array([1, 1, 0], dtype=complex) / np.sqrt(2),
        np.array([1, -1, 0], dtype=complex) / np.sqrt(2),
        np.array([1, 1j, 0], dtype=complex) / np.sqrt(2),
        np.array([1, -1j, 0], dtype=complex) / np.sqrt(2),
    ]
    
    X_gate = np.zeros((3, 3), dtype=complex)
    X_gate[0, 1] = 1.0
    X_gate[1, 0] = 1.0
    
    detunings = [-2*np.pi*0.002, 0.0, 2*np.pi*0.002]
    scales = [0.97, 1.0, 1.03]
    
    fidelities = []
    for det in detunings:
        for scl in scales:
            for init_state in target_states:
                rho = _simulate_pulse(pulse, det, scl)
                target = X_gate @ init_state
                f = _fidelity(rho, target)
                fidelities.append(f)
    
    fidelities = np.array(fidelities)
    infidelities = 1.0 - fidelities
    mean_inf = np.mean(infidelities)
    max_inf = np.max(infidelities)
    score = mean_inf + max_inf
    return score


def _enforce_constraints(pulse):
    """Enforce amplitude and slew rate constraints."""
    pulse = pulse.copy()
    
    # Amplitude bound
    for i in range(N_STEPS):
        amp = np.sqrt(pulse[i, 0]**2 + pulse[i, 1]**2)
        if amp > AMPLITUDE_BOUND:
            scale = AMPLITUDE_BOUND / amp
            pulse[i, 0] *= scale
            pulse[i, 1] *= scale
    
    # Slew rate bound
    prev = np.array([0.0, 0.0])
    for i in range(N_STEPS):
        delta = pulse[i] - prev
        norm = np.linalg.norm(delta) / DT
        if norm > SLEW_BOUND:
            pulse[i] = prev + SLEW_BOUND * DT * delta / (norm * DT)
        prev = pulse[i]
    
    # Final step to zero
    norm_final = np.linalg.norm(prev) / DT
    if norm_final > SLEW_BOUND:
        pulse[-1] = prev * (SLEW_BOUND * DT / np.linalg.norm(prev))
    
    return pulse


def _seed_pulse():
    """Generate DRAG-style Gaussian seed pulse."""
    t = np.arange(N_STEPS) * DT
    t_center = (N_STEPS - 1) * DT / 2.0
    sigma = 40.0 / 6.0
    
    envelope = np.exp(-0.5 * ((t - t_center) / sigma)**2)
    raw_area = np.sum(envelope) * DT
    envelope = envelope * (np.pi / raw_area)
    
    drag_coeff = -0.1
    envelope_deriv = np.gradient(envelope, DT)
    
    I_pulse = envelope
    Q_pulse = drag_coeff * envelope_deriv
    
    pulse = np.column_stack([I_pulse, Q_pulse])
    return _enforce_constraints(pulse)


def _param_pulse(params):
    """Generate pulse from parameters using spectral nulling representation.
    params: [amp_main, amp_drag, shift_main, shift_drag, sigma_scale]
    """
    amp_main, amp_drag, shift_main, shift_drag, sigma_scale = params
    t = np.arange(N_STEPS) * DT
    t_center = (N_STEPS - 1) * DT / 2.0
    sigma = (40.0 / 6.0) * sigma_scale
    
    # Main Gaussian envelope
    env_main = amp_main * np.exp(-0.5 * ((t - t_center - shift_main) / sigma)**2)
    raw_area_main = np.sum(env_main) * DT
    if raw_area_main > 1e-10:
        env_main = env_main * (np.pi / raw_area_main)
    
    # DRAG component: derivative-coupled Gaussian
    env_drag = amp_drag * np.exp(-0.5 * ((t - t_center - shift_drag) / sigma)**2)
    env_drag_deriv = np.gradient(env_drag, DT)
    
    I_pulse = env_main
    Q_pulse = env_drag_deriv
    
    pulse = np.column_stack([I_pulse, Q_pulse])
    return _enforce_constraints(pulse)


def _objective(params):
    if time.time() > _DEADLINE - 5:
        return 1.0
    p = _param_pulse(params)
    s = _compute_score(p)
    if s is None:
        return 1.0
    return s


def pulse(target_gate, hardware_model):
    """Main entry point."""
    if target_gate != 'X_pi':
        return np.zeros((N_STEPS, 2))
    
    # Start with seed
    best_pulse = _seed_pulse()
    best_score = _compute_score(best_pulse)
    if best_score is None:
        best_pulse = np.zeros((N_STEPS, 2))
        best_score = _compute_score(best_pulse)
        if best_score is None:
            best_score = 1.0
    
    # Spectral nulling optimization: optimize [amp_main, amp_drag, shift_main, shift_drag, sigma_scale]
    initial_guesses = [
        [0.3, -0.1, 0.0, 0.0, 1.0],
        [0.3, -0.15, 0.0, 0.5, 1.0],
        [0.3, -0.08, 0.0, -0.5, 1.0],
        [0.3, -0.1, 0.5, 0.0, 1.0],
        [0.3, -0.1, -0.5, 0.0, 1.0],
        [0.25, -0.12, 0.0, 0.0, 0.9],
        [0.35, -0.1, 0.0, 0.0, 1.1],
        [0.3, -0.2, 0.0, 0.0, 1.0],
        [0.3, -0.05, 0.0, 0.0, 1.0],
    ]
    
    for init in initial_guesses:
        if time.time() > _DEADLINE - 15:
            break
        try:
            result = minimize(_objective, init, method='Nelder-Mead',
                             options={'maxiter': 300, 'xatol': 1e-7, 'fatol': 1e-9})
            if result.fun < best_score:
                best_score = result.fun
                best_pulse = _param_pulse(result.x)
        except Exception:
            pass
    
    # Additional local refinement with tighter tolerances
    if time.time() < _DEADLINE - 30:
        try:
            result = minimize(_objective, result.x, method='Nelder-Mead',
                             options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
            if result.fun < best_score:
                best_score = result.fun
                best_pulse = _param_pulse(result.x)
        except Exception:
            pass
    
    # Try different DRAG strengths with the spectral nulling representation
    for drag_amp in [-0.25, -0.2, -0.15, -0.1, -0.05]:
        if time.time() > _DEADLINE - 15:
            break
        try:
            init = [0.3, drag_amp, 0.0, 0.0, 1.0]
            result = minimize(_objective, init, method='Nelder-Mead',
                             options={'maxiter': 200, 'xatol': 1e-7, 'fatol': 1e-9})
            if result.fun < best_score:
                best_score = result.fun
                best_pulse = _param_pulse(result.x)
        except Exception:
            pass
    
    return best_pulse
