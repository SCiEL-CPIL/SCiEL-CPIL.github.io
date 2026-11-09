import numpy as np
import time
from scipy.linalg import expm
from scipy.optimize import minimize, differential_evolution
from scipy.integrate import solve_ivp

def _precompute_operators():
    a = np.array([[0, 1, 0], [0, 0, np.sqrt(2)], [0, 0, 0]], dtype=complex)
    a_dag = a.conj().T
    n = a_dag @ a
    I3 = np.eye(3, dtype=complex)
    n_nm1 = n @ (n - I3)
    a_plus_adag = a + a_dag
    a_minus_adag = a - a_dag
    return a, a_dag, n, I3, n_nm1, a_plus_adag, a_minus_adag

def _simulate_unitary_fast(pulse_IQ, detuning, scale, anharmonicity, dt, n_steps,
                           a, a_dag, n, I3, n_nm1, a_plus_adag, a_minus_adag):
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

def _enforce_constraints(pulse_IQ, amp_bound, slew_bound, dt, n_steps):
    pulse = pulse_IQ.copy()
    amps = np.sqrt(pulse[:, 0]**2 + pulse[:, 1]**2)
    max_amp = np.max(amps)
    if max_amp > amp_bound:
        pulse *= (amp_bound / max_amp)
    
    max_slew = 0
    prev = np.array([0.0, 0.0])
    for i in range(n_steps):
        diff = np.linalg.norm(pulse[i] - prev)
        slew = diff / dt
        if slew > max_slew:
            max_slew = slew
        prev = pulse[i]
    diff = np.linalg.norm(np.array([0.0, 0.0]) - prev)
    slew = diff / dt
    if slew > max_slew:
        max_slew = slew
    if max_slew > slew_bound:
        pulse *= (slew_bound / max_slew)
    
    # Re-check amplitude after slew scaling
    amps = np.sqrt(pulse[:, 0]**2 + pulse[:, 1]**2)
    max_amp = np.max(amps)
    if max_amp > amp_bound:
        pulse *= (amp_bound / max_amp)
    
    return pulse

def _dop853_recheck(pulse_IQ, detuning, scale, anharmonicity, dt, n_steps):
    """Recheck nominal case using DOP853 for accuracy verification."""
    a = np.array([[0, 1, 0], [0, 0, np.sqrt(2)], [0, 0, 0]], dtype=complex)
    a_dag = a.conj().T
    n = a_dag @ a
    I3 = np.eye(3, dtype=complex)
    n_nm1 = n @ (n - I3)
    a_plus_adag = a + a_dag
    a_minus_adag = a - a_dag
    
    H_static = detuning * n + 0.5 * anharmonicity * n_nm1
    
    def hamiltonian(t, state):
        idx = int(t / dt)
        if idx >= n_steps:
            idx = n_steps - 1
        I_val = pulse_IQ[idx, 0]
        Q_val = pulse_IQ[idx, 1]
        H = H_static + 0.5 * scale * (I_val * a_plus_adag + 1j * Q_val * a_minus_adag)
        return -1j * H @ state
    
    T_total = n_steps * dt
    t_eval = np.linspace(0, T_total, 1000)
    
    X = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=complex)
    psi_0 = np.array([1, 0, 0], dtype=complex)
    psi_1 = np.array([0, 1, 0], dtype=complex)
    psi_01 = np.array([1, 1, 0], dtype=complex) / np.sqrt(2)
    psi_01n = np.array([1, -1, 0], dtype=complex) / np.sqrt(2)
    psi_0i1 = np.array([1, 1j, 0], dtype=complex) / np.sqrt(2)
    psi_0mi1 = np.array([1, -1j, 0], dtype=complex) / np.sqrt(2)
    initial_states = [psi_0, psi_1, psi_01, psi_01n, psi_0i1, psi_0mi1]
    
    infidelities = []
    for psi in initial_states:
        sol = solve_ivp(hamiltonian, [0, T_total], psi, method='DOP853',
                        rtol=1e-12, atol=1e-14, t_eval=[T_total])
        psi_final = sol.y[:, -1]
        phi = X @ psi
        F = np.abs(np.vdot(phi, psi_final))**2
        infid = 1.0 - F
        infidelities.append(infid)
    
    return np.mean(infidelities)

def pulse(target_gate, hardware_model):
    start_time = time.time()
    deadline = start_time + 320.0  # 10s safety margin before 330s kill

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
    t = t_samples - t_center

    detunings = [-detuning_bound, 0.0, detuning_bound]
    scales = [0.97, 1.0, 1.03]

    # Generate initial DRAG pulse with proper area = pi
    # For X-pi gate: integral of I(t) dt = pi (in rad)
    # Gaussian: A * sigma * sqrt(2*pi) = pi => A = 1/(sigma*sqrt(2*pi))
    sigma_seed = T_total / 6.0
    A_seed = 1.0 / (sigma_seed * np.sqrt(2 * np.pi))
    # Clip to amplitude bound
    if A_seed > amp_bound:
        A_seed = amp_bound
    
    beta_seed = 0.1  # DRAG coefficient for 3-level transmon
    I_base = A_seed * np.exp(-t**2 / (2 * sigma_seed**2))
    dI_dt = -A_seed * t / sigma_seed**2 * np.exp(-t**2 / (2 * sigma_seed**2))
    Q_drag = -beta_seed * dI_dt
    pulse_seed = np.column_stack([I_base, Q_drag])
    
    # Enforce constraints on seed
    pulse_seed = _enforce_constraints(pulse_seed, amp_bound, slew_bound, dt, n_steps)

    def _objective_direct(x_flat):
        p = x_flat.reshape(n_steps, 2)
        if not _check_validity(p, amp_bound, slew_bound, dt, n_steps):
            return 1e6
        scores = []
        for det in detunings:
            for sc in scales:
                U = _simulate_unitary_fast(p, det, sc, anharmonicity, dt, n_steps,
                                           a, a_dag, n, I3, n_nm1, a_plus_adag, a_minus_adag)
                score = _compute_score_from_unitary(U, X)
                scores.append(score)
        return np.mean(scores) + np.max(scores)

    def _objective_with_enforcement(x_flat):
        p = x_flat.reshape(n_steps, 2)
        p = _enforce_constraints(p, amp_bound, slew_bound, dt, n_steps)
        if not _check_validity(p, amp_bound, slew_bound, dt, n_steps):
            return 1e6
        scores = []
        for det in detunings:
            for sc in scales:
                U = _simulate_unitary_fast(p, det, sc, anharmonicity, dt, n_steps,
                                           a, a_dag, n, I3, n_nm1, a_plus_adag, a_minus_adag)
                score = _compute_score_from_unitary(U, X)
                scores.append(score)
        return np.mean(scores) + np.max(scores)

    best_pulse = pulse_seed.copy()
    best_score = _objective_direct(pulse_seed.flatten())
    
    # Stage 1: Nelder-Mead on full 160-dim space (robust, no gradient needed)
    if time.time() < deadline - 30.0:
        try:
            x0 = best_pulse.flatten()
            res_nm = minimize(_objective_with_enforcement, x0, method='Nelder-Mead',
                              options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8,
                                       'adaptive': True, 'maxfev': 500})
            if res_nm.fun < best_score:
                best_score = res_nm.fun
                p = res_nm.x.reshape(n_steps, 2)
                p = _enforce_constraints(p, amp_bound, slew_bound, dt, n_steps)
                if _check_validity(p, amp_bound, slew_bound, dt, n_steps):
                    best_pulse = p
                    best_score = _objective_direct(p.flatten())
        except Exception as e:
            pass

    # Stage 2: L-BFGS-B refinement
    if time.time() < deadline - 20.0:
        try:
            x0 = best_pulse.flatten()
            bounds_full = [(-0.39, 0.39)] * 160
            res_local = minimize(_objective_direct, x0, method='L-BFGS-B', bounds=bounds_full,
                                 options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8,
                                          'maxfun': 1000})
            if res_local.fun < best_score:
                best_score = res_local.fun
                p = res_local.x.reshape(n_steps, 2)
                p = _enforce_constraints(p, amp_bound, slew_bound, dt, n_steps)
                if _check_validity(p, amp_bound, slew_bound, dt, n_steps):
                    best_pulse = p
                    best_score = _objective_direct(p.flatten())
        except Exception as e:
            pass

    # Stage 3: Another Nelder-Mead pass if time remains
    if time.time() < deadline - 15.0:
        try:
            x0 = best_pulse.flatten()
            res_nm2 = minimize(_objective_with_enforcement, x0, method='Nelder-Mead',
                               options={'maxiter': 150, 'xatol': 1e-7, 'fatol': 1e-9,
                                        'adaptive': True, 'maxfev': 400})
            if res_nm2.fun < best_score:
                p = res_nm2.x.reshape(n_steps, 2)
                p = _enforce_constraints(p, amp_bound, slew_bound, dt, n_steps)
                if _check_validity(p, amp_bound, slew_bound, dt, n_steps):
                    s = _objective_direct(p.flatten())
                    if s < best_score:
                        best_pulse = p
                        best_score = s
        except Exception as e:
            pass

    # Final validity check
    if not _check_validity(best_pulse, amp_bound, slew_bound, dt, n_steps):
        best_pulse = _enforce_constraints(best_pulse, amp_bound, slew_bound, dt, n_steps)
        if not _check_validity(best_pulse, amp_bound, slew_bound, dt, n_steps):
            best_pulse = pulse_seed

    return best_pulse
