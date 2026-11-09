import numpy as np
from scipy.linalg import expm
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

def pulse(target_gate, hardware_model):
    start_time = time.time()
    deadline = start_time + 280.0

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

    def make_drag_pulse(sigma, beta):
        A = np.sqrt(np.pi / 2) / sigma
        I_base = A * np.exp(-(t_samples - t_center)**2 / (2 * sigma**2))
        dI_dt = -A * (t_samples - t_center) / sigma**2 * np.exp(-(t_samples - t_center)**2 / (2 * sigma**2))
        Q_drag = -beta * dI_dt
        return np.column_stack([I_base, Q_drag])

    # Seed pulse
    sigma_seed = T_total / 6.0
    pulse_seed = make_drag_pulse(sigma_seed, 0.0)
    best_pulse = pulse_seed.copy()
    best_score = evaluate_pulse(pulse_seed)

    # Coarse grid search
    sigma_range = np.linspace(T_total / 9.0, T_total / 4.5, 12)
    beta_range = np.linspace(-0.15, 0.15, 13)
    for sigma in sigma_range:
        if time.time() > deadline - 80.0:
            break
        for beta in beta_range:
            if time.time() > deadline - 80.0:
                break
            pulse_IQ = make_drag_pulse(sigma, beta)
            score = evaluate_pulse(pulse_IQ)
            if score < best_score:
                best_score = score
                best_pulse = pulse_IQ.copy()
                best_sigma, best_beta = sigma, beta

    # Fine grid search
    if time.time() < deadline - 40.0:
        s_lo = max(T_total / 10.0, best_sigma - 0.4)
        s_hi = min(T_total / 3.5, best_sigma + 0.4)
        sigma_fine = np.linspace(s_lo, s_hi, 10)
        beta_fine = np.linspace(best_beta - 0.03, best_beta + 0.03, 10)
        for sigma in sigma_fine:
            if time.time() > deadline - 40.0:
                break
            for beta in beta_fine:
                if time.time() > deadline - 40.0:
                    break
                pulse_IQ = make_drag_pulse(sigma, beta)
                score = evaluate_pulse(pulse_IQ)
                if score < best_score:
                    best_score = score
                    best_pulse = pulse_IQ.copy()
                    best_sigma, best_beta = sigma, beta

    # Local Nelder-Mead polish
    if time.time() < deadline - 15.0:
        from scipy.optimize import minimize
        def objective(params):
            p = make_drag_pulse(params[0], params[1])
            return evaluate_pulse(p)
        try:
            result = minimize(objective, [best_sigma, best_beta], method='Nelder-Mead',
                              options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-8})
            if result.fun < best_score:
                best_score = result.fun
                best_pulse = make_drag_pulse(result.x[0], result.x[1])
        except Exception:
            pass

    return best_pulse
