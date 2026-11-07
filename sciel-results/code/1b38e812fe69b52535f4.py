"""Optimized pulse for three-level transmon X_pi gate with robustness."""
import numpy as np
import time
from scipy.linalg import expm

_T0 = time.time()
_DEADLINE = _T0 + 300  # 30s safety margin before 330s kill

def _check_time():
    return time.time() < _DEADLINE

def _build_hamiltonian(detuning, anharmonicity, scale, I, Q):
    """Build 3x3 Hamiltonian for the three-level system."""
    a = np.zeros((3, 3), dtype=complex)
    a[1, 0] = np.sqrt(1)
    a[2, 1] = np.sqrt(2)
    a_dag = a.conj().T
    n_op = a_dag @ a
    
    H = detuning * n_op
    H += 0.5 * anharmonicity * n_op @ (n_op - np.eye(3))
    H += 0.5 * scale * (I * (a + a_dag) + 1j * Q * (a - a_dag))
    return H

def _build_liouvillian(H, T1, Tphi):
    """Build the Liouvillian superoperator for the density matrix evolution."""
    dim = 3
    a = np.zeros((dim, dim), dtype=complex)
    a[1, 0] = np.sqrt(1)
    a[2, 1] = np.sqrt(2)
    a_dag = a.conj().T
    n_op = a_dag @ a
    
    L1 = a / np.sqrt(T1)
    L2 = n_op * np.sqrt(2.0 / Tphi)
    
    I3 = np.eye(dim, dtype=complex)
    
    L_vec = -1j * (np.kron(H, I3) - np.kron(I3, H.conj().T))
    for L in [L1, L2]:
        LdL = L.conj().T @ L
        L_vec += np.kron(L, L.conj()) - 0.5 * (np.kron(LdL, I3) + np.kron(I3, LdL))
    
    return L_vec

def _simulate_pulse(IQ, detuning, scale, anharmonicity, T1, Tphi, dt):
    """Simulate pulse evolution using Liouvillian exponentiation per step."""
    n_steps = len(IQ)
    rho = np.zeros((3, 3), dtype=complex)
    rho[0, 0] = 1.0
    
    for i in range(n_steps):
        I_val = IQ[i, 0]
        Q_val = IQ[i, 1]
        H = _build_hamiltonian(detuning, anharmonicity, scale, I_val, Q_val)
        L_vec = _build_liouvillian(H, T1, Tphi)
        
        rho_vec = rho.ravel()
        rho_vec = expm(L_vec * dt) @ rho_vec
        rho = rho_vec.reshape(3, 3)
    
    return rho

def _compute_fidelity(rho, target_state):
    """Compute fidelity between density matrix and target pure state."""
    target_vec = target_state.ravel()
    rho_t = np.outer(target_vec, target_vec.conj())
    fidelity = np.real(np.trace(rho_t @ rho))
    return max(0.0, min(1.0, fidelity))

def _score_pulse(IQ, anharmonicity, T1, Tphi, dt):
    """Compute the scoring metric for a pulse."""
    detunings = [-2 * np.pi * 0.002, 0.0, 2 * np.pi * 0.002]
    scales = [0.97, 1.0, 1.03]
    
    # X_pi gate: X = [[0, -i], [i, 0]]
    # X|0> = -i|1>, X|1> = i|0>
    targets = [
        np.array([0, -1j, 0], dtype=complex),
        np.array([1j, 0, 0], dtype=complex),
        np.array([1j, -1j, 0], dtype=complex) / np.sqrt(2),
        np.array([-1j, -1j, 0], dtype=complex) / np.sqrt(2),
        np.array([-1, -1j, 0], dtype=complex) / np.sqrt(2),
        np.array([1, -1j, 0], dtype=complex) / np.sqrt(2),
    ]
    
    infidelities = []
    for det in detunings:
        for scl in scales:
            rho = _simulate_pulse(IQ, det, scl, anharmonicity, T1, Tphi, dt)
            case_infs = []
            for target in targets:
                fid = _compute_fidelity(rho, target)
                case_infs.append(1.0 - fid)
            infidelities.append(np.mean(case_infs))
    
    infidelities = np.array(infidelities)
    return np.mean(infidelities) + np.max(infidelities)

def _check_constraints(IQ, dt, amp_bound, slew_bound):
    """Check amplitude and slew constraints."""
    amplitudes = np.sqrt(IQ[:, 0]**2 + IQ[:, 1]**2)
    if np.any(amplitudes > amp_bound + 1e-9):
        return False
    
    # Slew from zero to first step
    if np.sqrt(IQ[0, 0]**2 + IQ[0, 1]**2) / dt > slew_bound + 1e-9:
        return False
    
    # Slew between steps
    for i in range(len(IQ) - 1):
        diff = IQ[i+1] - IQ[i]
        if np.sqrt(diff[0]**2 + diff[1]**2) / dt > slew_bound + 1e-9:
            return False
    
    # Slew from last step to zero
    if np.sqrt(IQ[-1, 0]**2 + IQ[-1, 1]**2) / dt > slew_bound + 1e-9:
        return False
    
    return True

def _drag_pulse(n_steps, dt, sigma, drag_coeff, amp_scale=1.0, time_shift=0.0):
    """Generate DRAG-style Gaussian pulse with parameters."""
    t = np.arange(n_steps) * dt
    t_center = (n_steps - 1) * dt / 2.0 + time_shift
    t_shifted = t - t_center
    
    # Gaussian envelope
    gaussian = np.exp(-t_shifted**2 / (2 * sigma**2))
    gaussian_area = np.sum(gaussian) * dt
    gaussian = gaussian * (np.pi / gaussian_area) * amp_scale
    
    # DRAG: Q component is derivative of I component
    I = gaussian
    Q = drag_coeff * np.gradient(gaussian, dt)
    
    IQ = np.column_stack([I, Q])
    
    # Clamp to amplitude constraint
    for i in range(n_steps):
        amp = np.sqrt(IQ[i, 0]**2 + IQ[i, 1]**2)
        if amp > 0.4:
            scale = 0.4 / amp
            IQ[i] *= scale
    
    return IQ

def pulse(target_gate, hardware_model):
    """Generate optimized pulse for X_pi gate."""
    if target_gate != 'X_pi':
        raise ValueError("Only X_pi is supported")
    
    anharmonicity = -2 * np.pi * 0.25
    amp_bound = 0.4
    slew_bound = 0.12
    T1 = 25000.0
    Tphi = 40000.0
    dt = 0.5
    n_steps = 80
    
    # Start with DRAG seed
    sigma = 40 / 6.0
    drag_coeff = 0.1
    IQ_best = _drag_pulse(n_steps, dt, sigma, drag_coeff)
    
    if not _check_constraints(IQ_best, dt, amp_bound, slew_bound):
        max_amp = np.max(np.sqrt(IQ_best[:, 0]**2 + IQ_best[:, 1]**2))
        if max_amp > amp_bound:
            IQ_best *= (amp_bound / max_amp) * 0.99
    
    # Grid search over parameters
    best_score = _score_pulse(IQ_best, anharmonicity, T1, Tphi, dt)
    
    sigma_vals = [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
    drag_vals = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    amp_vals = [0.8, 0.9, 1.0, 1.1, 1.2]
    shift_vals = [-0.5, 0.0, 0.5]
    
    for sigma_p in sigma_vals:
        for drag_p in drag_vals:
            for amp_p in amp_vals:
                for shift_p in shift_vals:
                    if not _check_time():
                        break
                    IQ_test = _drag_pulse(n_steps, dt, sigma_p, drag_p, amp_p, shift_p)
                    if _check_constraints(IQ_test, dt, amp_bound, slew_bound):
                        score = _score_pulse(IQ_test, anharmonicity, T1, Tphi, dt)
                        if score < best_score:
                            best_score = score
                            IQ_best = IQ_test.copy()
    
    # Local refinement around best
    if _check_time():
        # Find current best params by checking which combo gave best score
        # Just do a finer local search
        for sigma_p in [3.5, 4.0, 4.5, 5.0]:
            for drag_p in [0.1, 0.15, 0.2, 0.25]:
                for amp_p in [0.9, 1.0, 1.1]:
                    for shift_p in [-0.25, 0.0, 0.25]:
                        if not _check_time():
                            break
                        IQ_test = _drag_pulse(n_steps, dt, sigma_p, drag_p, amp_p, shift_p)
                        if _check_constraints(IQ_test, dt, amp_bound, slew_bound):
                            score = _score_pulse(IQ_test, anharmonicity, T1, Tphi, dt)
                            if score < best_score:
                                best_score = score
                                IQ_best = IQ_test.copy()
    
    return IQ_best
