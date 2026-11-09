import numpy as np
from scipy.linalg import expm
from scipy.integrate import solve_ivp
import time

def _precompute_operators():
    """Precompute all constant operators."""
    a = np.array([[0, 1, 0], [0, 0, np.sqrt(2)], [0, 0, 0]], dtype=complex)
    a_dag = a.conj().T
    n = a_dag @ a
    I3 = np.eye(3, dtype=complex)
    n_nm1 = n @ (n - I3)
    a_plus_adag = a + a_dag
    a_minus_adag = a - a_dag
    return a, a_dag, n, I3, n_nm1, a_plus_adag, a_minus_adag

def _simulate_unitary(pulse_IQ, detuning, scale, anharmonicity, dt, n_steps,
                      a, a_dag, n, I3, n_nm1, a_plus_adag, a_minus_adag):
    """Simulate the quantum dynamics and return the final unitary matrix."""
    U_total = I3.copy()
    H_static = detuning * n + 0.5 * anharmonicity * n_nm1
    for i in range(n_steps):
        I_val = pulse_IQ[i, 0]
        Q_val = pulse_IQ[i, 1]
        H = H_static + 0.5 * scale * (I_val * a_plus_adag + 1j * Q_val * a_minus_adag)
        U_step = expm(-1j * H * dt)
        U_total = U_step @ U_total
    return U_total

def _compute_score_from_unitary(U, X):
    """Compute the score (mean + max infidelity) from the final unitary."""
    psi_0 = np.array([1, 0, 0], dtype=complex)
    psi_1 = np.array([0, 1, 0], dtype=complex)
    psi_01 = np.array([1, 1, 0], dtype=complex) / np.sqrt(2)
    psi_01n = np.array([1, -1, 0], dtype=complex) / np.sqrt(2)
    psi_0i1 = np.array([1, 1j, 0], dtype=complex) / np.sqrt(2)
    psi_0mi1 = np.array([1, -1j, 0], dtype=complex) / np.sqrt(2)
    initial_states = [psi_0, psi_1, psi_01, psi_01n, psi_0i1, psi_0mi1]
    infidelities = []
    for psi in initial_states:
        phi = X @ psi
        psi_final = U @ psi
        F = np.abs(np.vdot(phi, psi_final))**2
        infid = 1.0 - F
        infidelities.append(infid)
    mean_infid = np.mean(infidelities)
    max_infid = np.max(infidelities)
    return mean_infid + max_infid

def _check_validity(pulse_IQ, amp_bound, slew_bound, dt, n_steps):
    """Check if pulse satisfies amplitude and slew rate constraints."""
    amplitudes = np.sqrt(pulse_IQ[:, 0]**2 + pulse_IQ[:, 1]**2)
    if np.any(amplitudes > amp_bound + 1e-10):
        return False
    prev = np.array([0.0, 0.0])
    for i in range(n_steps):
        diff = np.linalg.norm(pulse_IQ[i] - prev)
        if diff / dt > slew_bound + 1e-10:
            return False
        prev = pulse_IQ[i]
    diff = np.linalg.norm(np.array([0.0, 0.0]) - prev)
    if diff / dt > slew_bound + 1e-10:
        return False
    return True

def _enforce_constraints(pulse, amp_bound, slew_bound, dt, n_steps):
    """Scale pulse to satisfy amplitude and slew constraints."""
    amps = np.sqrt(pulse[:, 0]**2 + pulse[:, 1]**2)
    max_amp = np.max(amps)
    if max_amp > amp_bound:
        pulse = pulse * (amp_bound / max_amp)
    
    prev = np.array([0.0, 0.0])
    max_slew = 0.0
    for i in range(n_steps):
        diff = np.linalg.norm(pulse[i] - prev)
        slew = diff / dt
        max_slew = max(max_slew, slew)
        prev = pulse[i]
    diff = np.linalg.norm(-prev)
    max_slew = max(max_slew, diff / dt)
    
    if max_slew > slew_bound:
        pulse = pulse * (slew_bound / max_slew)
    
    return pulse

def pulse(target_gate, hardware_model):
    start_time = time.time()
    deadline = start_time + 315.0  # Safety margin before 330s kill

    dt = 0.5
    n_steps = 80
    T_total = 40.0
    anharmonicity = -2 * np.pi * 0.25
    amp_bound = 0.4
    slew_bound = 0.12
    detuning_bound = 2 * np.pi * 0.002

    a, a_dag, n, I3, n_nm1, a_plus_adag, a_minus_adag = _precompute_operators()
    X = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=complex)
    t_samples = (np.arange(n_steps) + 0.5) * dt
    t_center = T_total / 2.0

    detunings = [-detuning_bound, 0.0, detuning_bound]
    scales = [0.97, 1.0, 1.03]

    def evaluate_pulse(pulse_IQ):
        """Evaluate the full 9-case score for a pulse."""
        if not _check_validity(pulse_IQ, amp_bound, slew_bound, dt, n_steps):
            return float('inf')
        scores = []
        for det in detunings:
            for sc in scales:
                U = _simulate_unitary(pulse_IQ, det, sc, anharmonicity, dt, n_steps,
                                      a, a_dag, n, I3, n_nm1, a_plus_adag, a_minus_adag)
                score = _compute_score_from_unitary(U, X)
                scores.append(score)
        return np.mean(scores) + np.max(scores)

    # Build a richer basis for the pulse: Gaussian envelope modulated by
    # Chebyshev polynomials up to order 8, for both I and Q channels.
    # This gives 18 parameters (9 for I, 9 for Q) which is manageable for
    # gradient-based optimization.
    t_norm = (t_samples - t_center) / (T_total / 2.0)  # normalize to [-1, 1]
    envelope = np.exp(-0.5 * (t_samples - t_center)**2 / (T_total/5.0)**2)
    
    # Chebyshev polynomials T_0 through T_8
    def chebyshev(k, x):
        if k == 0:
            return np.ones_like(x)
        elif k == 1:
            return x.copy()
        else:
            Tkm2 = np.ones_like(x)
            Tkm1 = x.copy()
            for _ in range(2, k+1):
                Tk = 2*x*Tkm1 - Tkm2
                Tkm2 = Tkm1
                Tkm1 = Tk
            return Tkm1
    
    basis_funcs = []
    for k in range(9):
        T_k = chebyshev(k, t_norm)
        basis_funcs.append(envelope * T_k)
    
    # Build basis matrix: shape (n_steps, 18)
    # First 9 columns for I, last 9 for Q
    B_I = np.column_stack(basis_funcs)  # (80, 9)
    B_Q = np.column_stack(basis_funcs)  # (80, 9)
    
    # Create initial DRAG pulse to project onto this basis
    sigma_seed = T_total / 6.0
    A = np.sqrt(np.pi / 2) / sigma_seed
    I_seed = A * np.exp(-(t_samples - t_center)**2 / (2 * sigma_seed**2))
    dI_dt = -A * (t_samples - t_center) / sigma_seed**2 * np.exp(-(t_samples - t_center)**2 / (2 * sigma_seed**2))
    beta_seed = 0.1
    Q_seed = -beta_seed * dI_dt
    pulse_seed = np.column_stack([I_seed, Q_seed])
    pulse_seed = _enforce_constraints(pulse_seed, amp_bound, slew_bound, dt, n_steps)
    
    # Project seed onto basis
    cI_seed = np.linalg.lstsq(B_I, pulse_seed[:, 0], rcond=None)[0]
    cQ_seed = np.linalg.lstsq(B_Q, pulse_seed[:, 1], rcond=None)[0]
    init_coeffs = np.concatenate([cI_seed, cQ_seed])
    
    def make_pulse_from_coeffs(coeffs):
        """Reconstruct pulse from 18 basis coefficients."""
        I_p = B_I @ coeffs[:9]
        Q_p = B_Q @ coeffs[9:]
        p = np.column_stack([I_p, Q_p])
        return _enforce_constraints(p, amp_bound, slew_bound, dt, n_steps)
    
    best_pulse = make_pulse_from_coeffs(init_coeffs)
    best_score = evaluate_pulse(best_pulse)
    
    # Use L-BFGS-B for gradient-based optimization
    from scipy.optimize import minimize
    
    def objective(coeffs):
        if time.time() > deadline - 10:
            return float('inf')
        p = make_pulse_from_coeffs(coeffs)
        return evaluate_pulse(p)
    
    # First run: L-BFGS-B with moderate iterations
    if time.time() < deadline - 120:
        try:
            result = minimize(objective, init_coeffs, method='L-BFGS-B',
                             options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8,
                                      'maxls': 50})
            if result.fun < best_score:
                best_score = result.fun
                best_pulse = make_pulse_from_coeffs(result.x)
                init_coeffs = result.x.copy()
        except Exception:
            pass
    
    # Second run: finer L-BFGS-B from improved point
    if time.time() < deadline - 60:
        try:
            result = minimize(objective, init_coeffs, method='L-BFGS-B',
                             options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10,
                                      'maxls': 100})
            if result.fun < best_score:
                best_score = result.fun
                best_pulse = make_pulse_from_coeffs(result.x)
                init_coeffs = result.x.copy()
        except Exception:
            pass
    
    # Third run: even finer
    if time.time() < deadline - 20:
        try:
            result = minimize(objective, init_coeffs, method='L-BFGS-B',
                             options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12,
                                      'maxls': 200})
            if result.fun < best_score:
                best_score = result.fun
                best_pulse = make_pulse_from_coeffs(result.x)
        except Exception:
            pass
    
    # Final validity check
    if not _check_validity(best_pulse, amp_bound, slew_bound, dt, n_steps):
        # Fallback to seed if somehow invalid
        best_pulse = _enforce_constraints(pulse_seed, amp_bound, slew_bound, dt, n_steps)
    
    return best_pulse
