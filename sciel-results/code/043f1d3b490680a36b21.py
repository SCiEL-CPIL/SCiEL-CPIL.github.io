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
    deadline = start_time + 320.0

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
        amplitudes = np.sqrt(pulse_IQ[:, 0]**2 + pulse_IQ[:, 1]**2)
        amp_violation = np.sum(np.maximum(0, amplitudes - amp_bound)**2)
        
        scores = []
        for det in detunings:
            for sc in scales:
                U = _simulate_unitary(pulse_IQ, det, sc, anharmonicity, dt, n_steps,
                                      a, a_dag, n, I3, n_nm1, a_plus_adag, a_minus_adag)
                score = _compute_score_from_unitary(U, X)
                scores.append(score)
        
        raw_score = np.mean(scores) + np.max(scores)
        return raw_score + 100.0 * amp_violation

    def feedback_pulse(params):
        """
        Generate pulse using time-domain feedback control.
        
        The idea: we want the state to follow a reference trajectory
        |psi_ref(t)> = U_ref(t) |psi_0>, where U_ref(t) = exp(-i * X * pi * t/T / 2).
        
        At each step, we compute the error between the current state and the
        reference state, then generate a control pulse to correct it.
        
        params: [Kp, Ki, Kd, ref_scale]
          Kp: proportional gain
          Ki: integral gain  
          Kd: derivative gain
          ref_scale: scaling of the reference rotation speed
        """
        Kp, Ki, Kd, ref_scale = params
        
        # Reference trajectory: rotation about X axis
        # U_ref(t) = exp(-i * X * (pi * ref_scale * t/T_total) / 2)
        # For a pi gate, ref_scale=1 gives full rotation in T_total
        
        # Track state evolution under the current pulse
        psi = np.array([1, 0, 0], dtype=complex)
        pulse_IQ = np.zeros((n_steps, 2))
        
        # Integral of error
        integral_err = np.zeros(2)
        prev_err = np.zeros(2)
        
        H_static = 0.0 * n + 0.5 * anharmonicity * n_nm1
        
        for i in range(n_steps):
            t_now = t_samples[i]
            
            # Reference state at this time
            theta_ref = np.pi * ref_scale * t_now / T_total
            U_ref = expm(-1j * X * theta_ref)
            psi_ref = U_ref @ psi
            
            # Error: component of (psi_ref - psi) in the |1> direction (drive axis)
            # We want to project the error onto the drive direction
            # The drive couples |0> and |1>, so we look at the |1> component
            err_1 = psi_ref[1] - psi[1]
            err_0 = psi_ref[0] - psi[0]
            
            # The I component should drive population transfer |0> <-> |1>
            # The Q component should correct phase
            I_err = np.real(err_0)  # population transfer
            Q_err = np.imag(err_0)  # phase correction
            
            # PID control
            P = Kp * I_err
            I_term = Ki * integral_err[0]
            D = Kd * (I_err - prev_err[0]) / dt
            
            # For Q, use a simpler proportional control on phase error
            Q_term = Kp * 0.5 * Q_err
            
            pulse_IQ[i, 0] = P + I_term + D
            pulse_IQ[i, 1] = Q_term
            
            # Update integral
            integral_err[0] += I_err * dt
            integral_err[1] += Q_err * dt
            prev_err[0] = I_err
            
            # Simulate one step
            H = H_static + 0.5 * (pulse_IQ[i, 0] * a_plus_adag + 1j * pulse_IQ[i, 1] * a_minus_adag)
            U_step = expm(-1j * H * dt)
            psi = U_step @ psi
        
        # Enforce constraints
        amps = np.sqrt(pulse_IQ[:, 0]**2 + pulse_IQ[:, 1]**2)
        if np.max(amps) > amp_bound:
            pulse_IQ *= (amp_bound / np.max(amps))
        
        max_slew = 0
        prev = np.array([0.0, 0.0])
        for i in range(n_steps):
            diff = np.linalg.norm(pulse_IQ[i] - prev)
            slew = diff / dt
            if slew > max_slew:
                max_slew = slew
            prev = pulse_IQ[i]
        diff = np.linalg.norm(np.array([0.0, 0.0]) - prev)
        slew = diff / dt
        if slew > max_slew:
            max_slew = slew
        if max_slew > slew_bound:
            pulse_IQ *= (slew_bound / max_slew)
            
        return pulse_IQ

    def objective_feedback(params):
        p = feedback_pulse(params)
        return evaluate_pulse(p)

    # Initial guess
    best_params = np.array([1.0, 0.1, 0.01, 1.0])
    best_pulse = feedback_pulse(best_params)
    best_score = evaluate_pulse(best_pulse)
    
    # Optimize the 4 controller parameters
    if time.time() < deadline - 120.0:
        try:
            bounds = [(0.1, 5.0), (0.0, 2.0), (0.0, 0.5), (0.5, 1.5)]
            res = minimize(objective_feedback, best_params, method='Nelder-Mead',
                           bounds=bounds, options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
                best_pulse = feedback_pulse(res.x)
        except Exception:
            pass

    # Second stage: try different initial conditions and local search
    if time.time() < deadline - 60.0:
        try:
            # Try a few different starting points
            for trial_params in [np.array([2.0, 0.5, 0.05, 1.0]),
                                  np.array([0.5, 0.2, 0.02, 1.0]),
                                  np.array([1.5, 0.3, 0.03, 1.1]),
                                  np.array([0.8, 0.15, 0.01, 0.95])]:
                if time.time() > deadline - 30.0:
                    break
                p_trial = feedback_pulse(trial_params)
                s_trial = evaluate_pulse(p_trial)
                if s_trial < best_score:
                    best_score = s_trial
                    best_params = trial_params.copy()
                    best_pulse = p_trial
        except Exception:
            pass

    # Final refinement with L-BFGS-B on the 4 params
    if time.time() < deadline - 30.0:
        try:
            bounds = [(0.05, 10.0), (0.0, 5.0), (0.0, 1.0), (0.3, 2.0)]
            res = minimize(objective_feedback, best_params, method='L-BFGS-B',
                           bounds=bounds, options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
                best_pulse = feedback_pulse(res.x)
        except Exception:
            pass

    # Final validity check
    if not _check_validity(best_pulse, amp_bound, slew_bound, dt, n_steps):
        # Fallback: scale down to satisfy constraints
        amps = np.sqrt(best_pulse[:, 0]**2 + best_pulse[:, 1]**2)
        if np.max(amps) > amp_bound:
            best_pulse *= (amp_bound / np.max(amps))
        
        max_slew = 0
        prev = np.array([0.0, 0.0])
        for i in range(n_steps):
            diff = np.linalg.norm(best_pulse[i] - prev)
            slew = diff / dt
            if slew > max_slew:
                max_slew = slew
            prev = best_pulse[i]
        diff = np.linalg.norm(np.array([0.0, 0.0]) - prev)
        slew = diff / dt
        if slew > max_slew:
            max_slew = slew
        if max_slew > slew_bound:
            best_pulse *= (slew_bound / max_slew)
        
        if not _check_validity(best_pulse, amp_bound, slew_bound, dt, n_steps):
            # Last resort: zero pulse
            best_pulse = np.zeros((n_steps, 2))

    return best_pulse
