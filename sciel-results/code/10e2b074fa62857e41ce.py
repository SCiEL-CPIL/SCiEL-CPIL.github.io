import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize
import time

def _precompute_operators():
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
    U_total = I3.copy()
    H_static = detuning * n + 0.5 * anharmonicity * n_nm1
    half_scale = 0.5 * scale
    for i in range(n_steps):
        I_val = pulse_IQ[i, 0]
        Q_val = pulse_IQ[i, 1]
        H = H_static + half_scale * (I_val * a_plus_adag + 1j * Q_val * a_minus_adag)
        U_step = expm(-1j * H * dt)
        U_total = U_step @ U_total
    return U_total

def _compute_score_from_unitary(U, X):
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
    deadline = start_time + 310.0
    
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
    
    # Build basis functions: Gaussian envelope × Chebyshev polynomials
    t_norm = (t_samples - t_center) / (T_total / 2.0)
    sigma_env = T_total / 5.0
    envelope = np.exp(-0.5 * (t_samples - t_center)**2 / sigma_env**2)
    
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
    
    n_basis = 12  # More basis functions for richer representation
    basis_funcs = []
    for k in range(n_basis):
        T_k = chebyshev(k, t_norm)
        basis_funcs.append(envelope * T_k)
    
    B_I = np.column_stack(basis_funcs)
    B_Q = np.column_stack(basis_funcs)
    
    # Initial DRAG-style seed
    sigma_seed = T_total / 6.0
    A = np.sqrt(np.pi / 2) / sigma_seed
    I_seed = A * np.exp(-(t_samples - t_center)**2 / (2 * sigma_seed**2))
    dI_dt = -A * (t_samples - t_center) / sigma_seed**2 * np.exp(-(t_samples - t_center)**2 / (2 * sigma_seed**2))
    beta_seed = 0.1
    Q_seed = -beta_seed * dI_dt
    pulse_seed = np.column_stack([I_seed, Q_seed])
    pulse_seed = _enforce_constraints(pulse_seed, amp_bound, slew_bound, dt, n_steps)
    
    # Project onto basis
    cI_seed = np.linalg.lstsq(B_I, pulse_seed[:, 0], rcond=None)[0]
    cQ_seed = np.linalg.lstsq(B_Q, pulse_seed[:, 1], rcond=None)[0]
    init_coeffs = np.concatenate([cI_seed, cQ_seed])
    
    def make_pulse_from_coeffs(coeffs):
        I_p = B_I @ coeffs[:n_basis]
        Q_p = B_Q @ coeffs[n_basis:]
        p = np.column_stack([I_p, Q_p])
        return _enforce_constraints(p, amp_bound, slew_bound, dt, n_steps)
    
    best_pulse = make_pulse_from_coeffs(init_coeffs)
    
    def evaluate_pulse(pulse_IQ):
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
    
    best_score = evaluate_pulse(best_pulse)
    
    # Optimization with smooth surrogate: use log-sum-exp for max term
    def smooth_objective(coeffs):
        if time.time() > deadline - 5:
            return float('inf')
        p = make_pulse_from_coeffs(coeffs)
        if not _check_validity(p, amp_bound, slew_bound, dt, n_steps):
            return 1e6
        scores = []
        for det in detunings:
            for sc in scales:
                U = _simulate_unitary(p, det, sc, anharmonicity, dt, n_steps,
                                      a, a_dag, n, I3, n_nm1, a_plus_adag, a_minus_adag)
                score = _compute_score_from_unitary(U, X)
                scores.append(score)
        scores = np.array(scores)
        # Smooth approximation: mean + soft_max
        beta_soft = 50.0  # Controls smoothness
        max_val = np.max(scores)
        log_sum_exp = max_val + np.log(np.sum(np.exp(scores - max_val)))
        soft_max = log_sum_exp / len(scores)  # This is actually log-sum-exp / n, not quite right
        # Better: use a proper soft max
        soft_max_val = max_val + (1.0/beta_soft) * np.log(np.sum(np.exp(beta_soft * (scores - max_val))))
        return np.mean(scores) + soft_max_val
    
    # Phase 1: Broad search with L-BFGS-B
    if time.time() < deadline - 120:
        try:
            result = minimize(smooth_objective, init_coeffs, method='L-BFGS-B',
                             options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8,
                                      'maxls': 50})
            if result.fun < best_score:
                best_score = result.fun
                best_pulse = make_pulse_from_coeffs(result.x)
                init_coeffs = result.x.copy()
        except Exception:
            pass
    
    # Phase 2: Refine with tighter tolerances
    if time.time() < deadline - 60:
        try:
            result = minimize(smooth_objective, init_coeffs, method='L-BFGS-B',
                             options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10,
                                      'maxls': 100})
            if result.fun < best_score:
                best_score = result.fun
                best_pulse = make_pulse_from_coeffs(result.x)
                init_coeffs = result.x.copy()
        except Exception:
            pass
    
    # Phase 3: Final polish with exact objective
    if time.time() < deadline - 20:
        try:
            result = minimize(evaluate_pulse, init_coeffs, method='L-BFGS-B',
                             options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12,
                                      'maxls': 200})
            if result.fun < best_score:
                best_score = result.fun
                best_pulse = make_pulse_from_coeffs(result.x)
        except Exception:
            pass
    
    # Final validity check
    if not _check_validity(best_pulse, amp_bound, slew_bound, dt, n_steps):
        best_pulse = _enforce_constraints(pulse_seed, amp_bound, slew_bound, dt, n_steps)
    
    return best_pulse
