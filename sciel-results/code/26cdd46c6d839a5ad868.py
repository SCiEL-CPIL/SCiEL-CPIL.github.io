import numpy as np
from scipy.linalg import expm
from scipy.optimize import differential_evolution, minimize
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
    half_scale = 0.5 * scale
    for i in range(n_steps):
        I_val = pulse_IQ[i, 0]
        Q_val = pulse_IQ[i, 1]
        # H = H_static + half_scale * (I * (a + a†) + i Q * (a - a†))
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

def _generate_pulse(params, dt, n_steps):
    """
    Generate pulse from parameters:
    params = [A, sigma, beta, t_center, phi]
    A: Amplitude
    sigma: Gaussian width
    beta: DRAG coefficient
    t_center: Time center
    phi: Phase offset
    """
    A, sigma, beta, t_center, phi = params
    
    # Ensure sigma is positive
    sigma = abs(sigma) + 1e-6
    A = abs(A)
    
    t_samples = (np.arange(n_steps) + 0.5) * dt
    
    # Gaussian envelope
    gauss = np.exp(-(t_samples - t_center)**2 / (2 * sigma**2))
    
    # Derivative of Gaussian for DRAG
    d_gauss = - (t_samples - t_center) / sigma**2 * gauss
    
    # I and Q components
    I = A * gauss
    Q = beta * A * d_gauss
    
    # Apply phase
    I_p = I * np.cos(phi) - Q * np.sin(phi)
    Q_p = I * np.sin(phi) + Q * np.cos(phi)
    
    pulse = np.column_stack([I_p, Q_p])
    return pulse

def pulse(target_gate, hardware_model):
    start_time = time.time()
    deadline = start_time + 315.0  # Safety margin

    dt = 0.5
    n_steps = 80
    T_total = 40.0
    anharmonicity = -2 * np.pi * 0.25
    amp_bound = 0.4
    slew_bound = 0.12
    detuning_bound = 2 * np.pi * 0.002

    a, a_dag, n, I3, n_nm1, a_plus_adag, a_minus_adag = _precompute_operators()
    X = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=complex)
    
    detunings = [-detuning_bound, 0.0, detuning_bound]
    scales = [0.97, 1.0, 1.03]

    def evaluate_params(params):
        """Evaluate the score for a given set of pulse parameters."""
        p = _generate_pulse(params, dt, n_steps)
        p = _enforce_constraints(p, amp_bound, slew_bound, dt, n_steps)
        
        if not _check_validity(p, amp_bound, slew_bound, dt, n_steps):
            return 1e6 # Penalty for invalid pulse (though enforce_constraints should prevent this)

        scores = []
        for det in detunings:
            for sc in scales:
                U = _simulate_unitary(p, det, sc, anharmonicity, dt, n_steps,
                                      a, a_dag, n, I3, n_nm1, a_plus_adag, a_minus_adag)
                score = _compute_score_from_unitary(U, X)
                scores.append(score)
        
        return np.mean(scores) + np.max(scores)

    # Initial guess for parameters
    # A ~ pi / (sqrt(pi) * sigma) for area pi. If sigma ~ 5ns, A ~ 0.35
    # sigma ~ 5ns (10 steps)
    # beta ~ 0.1 (typical DRAG)
    # t_center ~ 20ns
    # phi ~ 0
    initial_params = [0.35, 5.0, 0.1, 20.0, 0.0]
    
    # Bounds for parameters
    # A: [0, 0.5]
    # sigma: [2, 10]
    # beta: [-0.5, 0.5]
    # t_center: [15, 25]
    # phi: [-pi/4, pi/4]
    bounds = [
        (0.0, 0.5),
        (2.0, 10.0),
        (-0.5, 0.5),
        (15.0, 25.0),
        (-np.pi/4, np.pi/4)
    ]

    best_params = initial_params
    best_score = evaluate_params(initial_params)
    
    # Phase 1: Differential Evolution
    # Use a small population and few iterations to stay within time limits
    # maxiter=30, popsize=10 -> 300 evaluations.
    # Each eval is ~9 simulations. 2700 simulations.
    # Simulation is fast (~ms), so this is feasible.
    
    if time.time() < deadline - 150:
        try:
            de_result = differential_evolution(evaluate_params, bounds, 
                                               maxiter=40, 
                                               popsize=15, 
                                               seed=42,
                                               tol=1e-6,
                                               polish=False) # Disable polish to save time, we do it manually
            if de_result.fun < best_score:
                best_score = de_result.fun
                best_params = de_result.x
        except Exception:
            pass

    # Phase 2: Local Refinement with L-BFGS-B
    # Use the best params from DE as starting point
    if time.time() < deadline - 30:
        try:
            res = minimize(evaluate_params, best_params, method='L-BFGS-B',
                           options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except Exception:
            pass

    # Phase 3: Nelder-Mead for final polish (robust to non-smoothness of max term)
    if time.time() < deadline - 10:
        try:
            res_nm = minimize(evaluate_params, best_params, method='Nelder-Mead',
                              options={'maxiter': 50, 'xatol': 1e-6, 'fatol': 1e-8})
            if res_nm.fun < best_score:
                best_score = res_nm.fun
                best_params = res_nm.x
        except Exception:
            pass

    # Generate final pulse
    final_pulse = _generate_pulse(best_params, dt, n_steps)
    final_pulse = _enforce_constraints(final_pulse, amp_bound, slew_bound, dt, n_steps)
    
    # Final validity check
    if not _check_validity(final_pulse, amp_bound, slew_bound, dt, n_steps):
        # Fallback to a very simple valid pulse if something went wrong
        # A simple Gaussian with low amplitude
        t_samples = (np.arange(n_steps) + 0.5) * dt
        t_center = 20.0
        sigma = 5.0
        gauss = np.exp(-(t_samples - t_center)**2 / (2 * sigma**2))
        I = 0.3 * gauss
        Q = np.zeros_like(I)
        final_pulse = np.column_stack([I, Q])
        final_pulse = _enforce_constraints(final_pulse, amp_bound, slew_bound, dt, n_steps)

    return final_pulse
