import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize, differential_evolution
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

def _simulate_unitary_fast(pulse_IQ, detuning, scale, anharmonicity, dt, n_steps,
                      a, a_dag, n, I3, n_nm1, a_plus_adag, a_minus_adag):
    """Simulate the quantum dynamics and return the final unitary matrix."""
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

def _project_to_valid(pulse_IQ, amp_bound, slew_bound, dt, n_steps):
    """Project pulse to satisfy constraints by scaling."""
    p = pulse_IQ.copy()
    # Amplitude constraint
    amps = np.sqrt(p[:, 0]**2 + p[:, 1]**2)
    max_amp = np.max(amps)
    if max_amp > amp_bound:
        p *= amp_bound / max_amp
    # Slew constraint
    prev = np.array([0.0, 0.0])
    max_slew = 0
    for i in range(n_steps):
        diff = np.linalg.norm(p[i] - prev)
        slew = diff / dt
        max_slew = max(max_slew, slew)
        prev = p[i]
    diff = np.linalg.norm(-prev)
    max_slew = max(max_slew, diff / dt)
    if max_slew > slew_bound:
        p *= slew_bound / max_slew
    return p

def pulse(target_gate, hardware_model):
    start_time = time.time()
    deadline = start_time + 310.0  # Safety margin

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
                U = _simulate_unitary_fast(pulse_IQ, det, sc, anharmonicity, dt, n_steps,
                                      a, a_dag, n, I3, n_nm1, a_plus_adag, a_minus_adag)
                score = _compute_score_from_unitary(U, X)
                scores.append(score)
        return np.mean(scores) + np.max(scores)

    def make_drag_pulse(sigma, beta):
        """Create a DRAG-style Gaussian pulse."""
        A = np.sqrt(np.pi / 2) / sigma
        I_base = A * np.exp(-(t_samples - t_center)**2 / (2 * sigma**2))
        dI_dt = -A * (t_samples - t_center) / sigma**2 * np.exp(-(t_samples - t_center)**2 / (2 * sigma**2))
        Q_drag = -beta * dI_dt
        p = np.column_stack([I_base, Q_drag])
        return _project_to_valid(p, amp_bound, slew_bound, dt, n_steps)

    def make_parametric_pulse(params):
        """
        Create a parametric pulse from 5 parameters.
        params: [sigma, beta, t_center, amp_scale, q_offset]
        """
        sigma = params[0]
        beta = params[1]
        t_c = params[2]
        amp_scale = params[3]
        q_offset = params[4]
        
        # Base Gaussian with area proportional to amp_scale
        A = np.sqrt(np.pi / 2) * amp_scale / sigma
        I_base = A * np.exp(-(t_samples - t_c)**2 / (2 * sigma**2))
        dI_dt = -A * (t_samples - t_c) / sigma**2 * np.exp(-(t_samples - t_c)**2 / (2 * sigma**2))
        Q_drag = -beta * dI_dt + q_offset * np.exp(-(t_samples - t_c)**2 / (2 * sigma**2))
        p = np.column_stack([I_base, Q_drag])
        return _project_to_valid(p, amp_bound, slew_bound, dt, n_steps)

    # Initial seed: DRAG with physics-informed beta
    sigma_seed = T_total / 6.0
    beta_seed = 0.15
    pulse_seed = make_drag_pulse(sigma_seed, beta_seed)
    best_pulse = pulse_seed.copy()
    best_score = evaluate_pulse(pulse_seed)

    # Phase 1: Differential Evolution on 5-parameter DRAG family
    # This reduces the optimization space from 160 to 5 dimensions,
    # enabling global optimization to escape local minima
    if time.time() < deadline - 150:
        def objective_parametric(params):
            if time.time() > deadline - 130:
                return float('inf')
            p = make_parametric_pulse(params)
            return evaluate_pulse(p)
        
        bounds = [
            (T_total/10, T_total/4.5),   # sigma
            (-0.3, 0.4),                 # beta
            (T_total*0.3, T_total*0.7),  # t_center
            (0.5, 1.5),                  # amp_scale
            (-0.05, 0.05),               # q_offset
        ]
        
        try:
            result_de = differential_evolution(
                objective_parametric, bounds,
                seed=42, maxiter=30, popsize=15,
                tol=1e-6, polish=True
            )
            if result_de.fun < best_score:
                p_de = make_parametric_pulse(result_de.x)
                s_de = evaluate_pulse(p_de)
                if s_de < best_score:
                    best_score = s_de
                    best_pulse = p_de.copy()
        except Exception:
            pass

    # Phase 2: Coarse grid search to verify and potentially improve
    sigma_vals = np.linspace(T_total / 8, T_total / 4.5, 6)
    beta_vals = np.linspace(-0.2, 0.3, 8)
    
    for sigma in sigma_vals:
        if time.time() > deadline - 80:
            break
        for beta in beta_vals:
            if time.time() > deadline - 80:
                break
            p = make_drag_pulse(sigma, beta)
            s = evaluate_pulse(p)
            if s < best_score:
                best_score = s
                best_pulse = p.copy()

    # Phase 3: Direct time-domain optimization of all 160 parameters
    if time.time() < deadline - 60:
        def objective_flat(flat_params):
            if time.time() > deadline - 20:
                return float('inf')
            p = flat_params.reshape(n_steps, 2)
            p = _project_to_valid(p, amp_bound, slew_bound, dt, n_steps)
            return evaluate_pulse(p)
        
        init_flat = best_pulse.flatten()
        try:
            result = minimize(objective_flat, init_flat, method='L-BFGS-B',
                             options={'maxiter': 40, 'ftol': 1e-12, 'gtol': 1e-8,
                                      'maxls': 40})
            if result.fun < best_score:
                p_opt = _project_to_valid(result.x.reshape(n_steps, 2), 
                                          amp_bound, slew_bound, dt, n_steps)
                s_opt = evaluate_pulse(p_opt)
                if s_opt < best_score:
                    best_score = s_opt
                    best_pulse = p_opt.copy()
        except Exception:
            pass

    # Phase 4: Second round of refinement
    if time.time() < deadline - 20:
        def objective_flat2(flat_params):
            if time.time() > deadline - 10:
                return float('inf')
            p = flat_params.reshape(n_steps, 2)
            p = _project_to_valid(p, amp_bound, slew_bound, dt, n_steps)
            return evaluate_pulse(p)
        
        init_flat2 = best_pulse.flatten()
        try:
            result2 = minimize(objective_flat2, init_flat2, method='L-BFGS-B',
                              options={'maxiter': 20, 'ftol': 1e-14, 'gtol': 1e-10,
                                       'maxls': 20})
            if result2.fun < best_score:
                p_opt2 = _project_to_valid(result2.x.reshape(n_steps, 2), 
                                           amp_bound, slew_bound, dt, n_steps)
                s_opt2 = evaluate_pulse(p_opt2)
                if s_opt2 < best_score:
                    best_score = s_opt2
                    best_pulse = p_opt2.copy()
        except Exception:
            pass

    # Final validity check and projection
    best_pulse = _project_to_valid(best_pulse, amp_bound, slew_bound, dt, n_steps)
    
    return best_pulse
