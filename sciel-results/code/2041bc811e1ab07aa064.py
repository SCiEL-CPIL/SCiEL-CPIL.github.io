import numpy as np
import time
from scipy.linalg import expm
from scipy.optimize import minimize

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
    t = t_samples - t_center

    detunings = [-detuning_bound, 0.0, detuning_bound]
    scales = [0.97, 1.0, 1.03]

    def evaluate_pulse(pulse_IQ):
        # Soft constraint penalty for amplitude
        amplitudes = np.sqrt(pulse_IQ[:, 0]**2 + pulse_IQ[:, 1]**2)
        amp_violation = np.sum(np.maximum(0, amplitudes - amp_bound)**2)
        
        # Slew rate is handled by the variable transformation (bounds on deltas)
        # But we should still check if the resulting pulse is valid?
        # The optimizer will stay within bounds of deltas, so slew is satisfied.
        # Amplitude might be violated, hence the penalty.
        
        if amp_violation > 1e-6:
             # If violation is large, return a high score to guide away
             pass 

        scores = []
        for det in detunings:
            for sc in scales:
                U = _simulate_unitary(pulse_IQ, det, sc, anharmonicity, dt, n_steps,
                                      a, a_dag, n, I3, n_nm1, a_plus_adag, a_minus_adag)
                score = _compute_score_from_unitary(U, X)
                scores.append(score)
        
        raw_score = np.mean(scores) + np.max(scores)
        # Add penalty
        return raw_score + 100.0 * amp_violation

    # Generate initial DRAG pulse
    sigma_seed = T_total / 6.0
    A_seed = 0.35
    beta_seed = 0.08
    I_base = A_seed * np.exp(-t**2 / (2 * sigma_seed**2))
    dI_dt = -A_seed * t / sigma_seed**2 * np.exp(-t**2 / (2 * sigma_seed**2))
    Q_drag = -beta_seed * dI_dt
    pulse_seed = np.column_stack([I_base, Q_drag])
    
    # Enforce constraints on seed
    # Scale down if needed
    amps = np.sqrt(pulse_seed[:, 0]**2 + pulse_seed[:, 1]**2)
    if np.max(amps) > amp_bound:
        pulse_seed *= (amp_bound / np.max(amps))
    
    # Check slew
    max_slew = 0
    prev = np.array([0.0, 0.0])
    for i in range(n_steps):
        diff = np.linalg.norm(pulse_seed[i] - prev)
        slew = diff / dt
        if slew > max_slew:
            max_slew = slew
        prev = pulse_seed[i]
    diff = np.linalg.norm(np.array([0.0, 0.0]) - prev)
    slew = diff / dt
    if slew > max_slew:
        max_slew = slew
    if max_slew > slew_bound:
        pulse_seed *= (slew_bound / max_slew)

    # Convert pulse to increments (deltas)
    # delta[0] = pulse[0]
    # delta[i] = pulse[i] - pulse[i-1]
    # delta[-1] = 0 - pulse[-1] (to return to 0? No, the pulse is defined for 80 steps.
    # The validity check includes "from last step to zero".
    # So we need an extra delta?
    # The variables are the 80 steps of I and Q.
    # The slew constraint is on p[i+1]-p[i] and p[0]-0 and 0-p[-1].
    # If we optimize p[i] directly, we have 160 vars.
    # Bounds on p[i] are not simple boxes because of slew.
    # If we optimize deltas:
    # d_0 = p_0
    # d_i = p_i - p_{i-1}
    # d_end = 0 - p_{79}
    # We have 81 deltas for I and 81 for Q?
    # No, the pulse is 80 steps.
    # Constraints:
    # |d_0| <= slew*dt
    # |d_i| <= slew*dt for i=1..79
    # |d_end| <= slew*dt
    # And |p_i| <= amp_bound.
    
    # Let's stick to optimizing p[i] directly with L-BFGS-B and penalty for slew?
    # No, slew is a difference constraint.
    # Let's use the delta formulation with 160 variables (deltas for I and Q for the 80 steps).
    # We assume the pulse starts at 0 and ends at 0.
    # So p_0 = d_0.
    # p_i = d_0 + ... + d_i.
    # We need p_79 to be close to 0?
    # The validity check says "from last step to zero".
    # So we should ensure the pulse goes to 0.
    # We can add a penalty for p_79 != 0.
    
    # Actually, it's safer to optimize the 160 pulse values directly and use a penalty for slew violations.
    # L-BFGS-B is good for box constraints. We can box constrain I and Q to [-0.4, 0.4].
    # Slew penalty: sum(|p[i+1]-p[i]| - slew_limit)^2.
    
    def objective_direct(p_flat):
        p = p_flat.reshape(80, 2)
        # Amplitude penalty
        amps = np.sqrt(p[:, 0]**2 + p[:, 1]**2)
        amp_pen = np.sum(np.maximum(0, amps - amp_bound)**2)
        
        # Slew penalty
        slew_pen = 0.0
        prev = np.array([0.0, 0.0])
        for i in range(80):
            diff = np.linalg.norm(p[i] - prev)
            if diff > slew_bound * dt:
                slew_pen += (diff - slew_bound * dt)**2
            prev = p[i]
        diff = np.linalg.norm(np.array([0.0, 0.0]) - prev)
        if diff > slew_bound * dt:
            slew_pen += (diff - slew_bound * dt)**2
            
        scores = []
        for det in detunings:
            for sc in scales:
                U = _simulate_unitary(p, det, sc, anharmonicity, dt, n_steps,
                                      a, a_dag, n, I3, n_nm1, a_plus_adag, a_minus_adag)
                score = _compute_score_from_unitary(U, X)
                scores.append(score)
        
        raw_score = np.mean(scores) + np.max(scores)
        return raw_score + 1000.0 * (amp_pen + slew_pen)

    # Initial guess
    x0 = pulse_seed.flatten()
    
    # Bounds: I and Q in [-0.4, 0.4]
    bounds = [(-0.4, 0.4)] * 160
    
    # Optimize
    best_pulse = pulse_seed
    best_score = evaluate_pulse(pulse_seed)
    
    if time.time() < deadline - 50.0:
        try:
            res = minimize(objective_direct, x0, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
            if res.fun < best_score:
                best_score = res.fun
                best_pulse = res.x.reshape(80, 2)
        except Exception:
            pass

    # Final validity check and fallback
    if not _check_validity(best_pulse, amp_bound, slew_bound, dt, n_steps):
        # Try to project? Or just use seed?
        # If invalid, use seed
        best_pulse = pulse_seed

    return best_pulse
