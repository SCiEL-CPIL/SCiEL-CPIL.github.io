"""Optimized pulse for three-level transmon X_pi gate with robustness."""
import numpy as np
import time
from scipy.linalg import expm
from scipy.optimize import minimize

_DEADLINE = time.time() + 310  # safety margin before 330s kill

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

_I3 = np.eye(3)


def _build_liouvillian(detuning, scale, I, Q):
    """Build the Liouvillian superoperator for one step."""
    H = detuning * _n_op
    H += 0.5 * ANHARMONICITY * (_n_op @ (_n_op - _I3))
    H += 0.5 * scale * (I * (_a + _a_dag) + 1j * Q * (_a - _a_dag))
    
    H_t = H.T
    L_ham = -1j * (np.kron(_I3, H_t) - np.kron(H_t, _I3))
    
    L_diss = np.zeros((9, 9), dtype=complex)
    for Lk, Lk_dag in [(_L1, _L1_dag), (_L2, _L2_dag)]:
        Lk_dag_Lk = Lk_dag @ Lk
        L_diss += np.kron(Lk_dag.T, Lk) - 0.5 * (np.kron(_I3, Lk_dag_Lk.T) + np.kron(Lk_dag_Lk, _I3))
    
    return L_ham + L_diss


def _simulate_pulse(pulse, detuning, scale):
    """Simulate the pulse and return the final density matrix."""
    rho = np.zeros((3, 3), dtype=complex)
    rho[0, 0] = 1.0
    
    for i in range(N_STEPS):
        I_val = pulse[i, 0]
        Q_val = pulse[i, 1]
        L = _build_liouvillian(detuning, scale, I_val, Q_val)
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
    """Generate pulse from parameters: [amp, drag, sigma, q_offset, i_offset]."""
    amp, drag, sigma, q_offset, i_offset = params
    t = np.arange(N_STEPS) * DT
    t_center = (N_STEPS - 1) * DT / 2.0
    
    envelope = amp * np.exp(-0.5 * ((t - t_center) / sigma)**2)
    raw_area = np.sum(envelope) * DT
    if raw_area > 1e-10:
        envelope = envelope * (np.pi / raw_area)
    
    envelope_deriv = np.gradient(envelope, DT)
    
    I_pulse = envelope + i_offset
    Q_pulse = drag * envelope_deriv + q_offset
    
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
    
    # Try multiple optimization strategies with different initial guesses
    initial_guesses = [
        [0.3, -0.1, 40.0/6.0, 0.0, 0.0],
        [0.25, -0.08, 40.0/6.0, 0.0, 0.0],
        [0.35, -0.12, 40.0/6.0, 0.0, 0.0],
        [0.3, -0.05, 40.0/6.0, 0.0, 0.0],
        [0.28, -0.15, 40.0/6.0, 0.0, 0.0],
        [0.32, -0.03, 40.0/6.0, 0.0, 0.0],
        [0.3, -0.1, 45.0/6.0, 0.0, 0.0],
        [0.3, -0.1, 35.0/6.0, 0.0, 0.0],
        [0.3, 0.0, 40.0/6.0, 0.0, 0.0],
        [0.3, 0.05, 40.0/6.0, 0.0, 0.0],
    ]
    
    for init in initial_guesses:
        if time.time() > _DEADLINE - 10:
            break
        try:
            result = minimize(_objective, init, method='Nelder-Mead',
                             options={'maxiter': 120, 'xatol': 1e-5, 'fatol': 1e-7})
            if result.fun < best_score:
                best_score = result.fun
                best_pulse = _param_pulse(result.x)
        except Exception:
            pass
    
    # Try a few more with different DRAG strengths and offsets
    for drag in [-0.2, -0.15, -0.1, -0.05, 0.0, 0.05]:
        if time.time() > _DEADLINE - 10:
            break
        try:
            init = [0.3, drag, 40.0/6.0, 0.0, 0.0]
            result = minimize(_objective, init, method='Nelder-Mead',
                             options={'maxiter': 80, 'xatol': 1e-5, 'fatol': 1e-7})
            if result.fun < best_score:
                best_score = result.fun
                best_pulse = _param_pulse(result.x)
        except Exception:
            pass
    
    return best_pulse
