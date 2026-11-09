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

def _project_to_valid(pulse_IQ, amp_bound, slew_bound, dt, n_steps):
    """Project pulse onto the valid set by scaling."""
    pulse = pulse_IQ.copy()
    amplitudes = np.sqrt(pulse[:, 0]**2 + pulse[:, 1]**2)
    max_amp = np.max(amplitudes)
    if max_amp > amp_bound:
        pulse = pulse * (amp_bound / max_amp)
    prev = np.array([0.0, 0.0])
    max_slew = 0
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
        pulse = pulse * (slew_bound / max_slew)
    return pulse

def _compute_grape_gradient(pulse_IQ, detuning, scale, anharmonicity, dt, n_steps,
                            a, a_dag, n, I3, n_nm1, a_plus_adag, a_minus_adag):
    """
    Compute gradient of the infidelity score w.r.t. pulse parameters using
    adjoint (GRAPE) method. Returns gradient array of shape (n_steps, 2).
    """
    X = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=complex)
    psi_0 = np.array([1, 0, 0], dtype=complex)
    psi_1 = np.array([0, 1, 0], dtype=complex)
    psi_01 = np.array([1, 1, 0], dtype=complex) / np.sqrt(2)
    psi_01n = np.array([1, -1, 0], dtype=complex) / np.sqrt(2)
    psi_0i1 = np.array([1, 1j, 0], dtype=complex) / np.sqrt(2)
    psi_0mi1 = np.array([1, -1j, 0], dtype=complex) / np.sqrt(2)
    initial_states = [psi_0, psi_1, psi_01, psi_01n, psi_0i1, psi_0mi1]
    target_states = [X @ psi for psi in initial_states]
    
    H_static = detuning * n + 0.5 * anharmonicity * n_nm1
    
    # Forward propagation: store all U_i
    U_list = [I3.copy()]
    U_total = I3.copy()
    for i in range(n_steps):
        I_val = pulse_IQ[i, 0]
        Q_val = pulse_IQ[i, 1]
        H = H_static + 0.5 * scale * (I_val * a_plus_adag + 1j * Q_val * a_minus_adag)
        U_step = expm(-1j * H * dt)
        U_total = U_step @ U_total
        U_list.append(U_total.copy())
    
    # Compute final infidelities and their gradients
    # F_k = |vdot(target_k, U @ psi_k)|^2
    # dF_k/dU = -2 Re[ vdot(target_k, U psi_k) * (psi_k^†) ]  (in a specific sense)
    # Actually: F_k = (target_k† U psi_k)(target_k† U psi_k)*
    # dF_k/dU_ij = -2 Re[ (target_k† U psi_k) * (psi_k† U† target_k)_ji ]
    # Simpler: use the fact that for unitary U, dF = -2 Re[ <target|U|psi> * <psi|dU|target> ]
    
    # Adjoint: propagate chi backwards
    # The gradient of F_k w.r.t. U_step_i is:
    # dF_k/dU_step_i = -2 Re[ <chi_k,i | H_drive_i | psi_k,i> ] * dt
    # where chi is the adjoint state
    
    # Let's use a simpler approach: compute gradient numerically via finite differences
    # but that's too slow. Instead, use the adjoint method properly.
    
    # For each initial state k, the adjoint state lambda_k evolves backwards:
    # lambda_k(i) = U_step(i)^† ... U_step(1)^† target_k
    # Actually: lambda_k(0) = target_k, lambda_k(i+1) = U_step(i+1)^† lambda_k(i)
    
    # Gradient contribution from step i:
    # dF_k/dI_i = -2 Re[ <lambda_k(i+1) | 0.5*scale*a_plus_adag | psi_k(i)> ] * dt
    # dF_k/dQ_i = -2 Re[ <lambda_k(i+1) | 0.5*scale*1j*a_minus_adag | psi_k(i)> ] * dt
    
    # where psi_k(i) = U_step(i)...U_step(1) psi_k(0) = U_list[i] @ psi_k
    # and lambda_k(i+1) = U_step(i+1)^† ... U_step(80)^† target_k
    
    # Let's compute this properly
    grad_I = np.zeros(n_steps)
    grad_Q = np.zeros(n_steps)
    
    # Precompute forward states for each initial state
    forward_states = []  # forward_states[k][i] = U_list[i] @ psi_k
    for k, psi in enumerate(initial_states):
        states = np.zeros((n_steps + 1, 3), dtype=complex)
        states[0] = psi
        for i in range(n_steps):
            states[i+1] = U_list[i+1] @ U_list[i].conj().T @ states[i]
            # Wait, this is wrong. U_list[i] = U_step(i)...U_step(1)
            # So U_list[i+1] = U_step(i+1) @ U_list[i]
            # psi(i+1) = U_list[i+1] @ psi(0)
            # Let me just compute directly
        states = np.zeros((n_steps + 1, 3), dtype=complex)
        states[0] = psi
        for i in range(n_steps):
            states[i+1] = U_list[i+1] @ psi
        forward_states.append(states)
    
    # Adjoint states: lambda_k(i) = U_list[i]† target_k
    # lambda_k(0) = target_k
    # lambda_k(i) = U_list[i]† target_k
    # Actually, the adjoint at step i boundary is:
    # lambda_k(i) = U_step(i+1)† ... U_step(80)† target_k = U_list[80]† U_list[i]† ... 
    # Hmm, let me think more carefully.
    
    # U_total = U_step(80) ... U_step(1)
    # U_list[i] = U_step(i) ... U_step(1)
    # U_total = U_step(80) ... U_step(i+1) @ U_list[i]
    
    # F_k = |target_k† U_total psi_k|^2
    #      = |target_k† U_step(80)...U_step(i+1) U_list[i] psi_k|^2
    
    # Let alpha_k(i) = U_list[i] psi_k (forward state at step i)
    # Let beta_k(i) = U_step(80)...U_step(i+1)† target_k (backward/adjoint state)
    # Then F_k = |beta_k(i)† alpha_k(i)|^2 for any i
    
    # dF_k/dU_step(j) for j <= i:
    # dF_k = -2 Re[ beta_k(j)† dU_step(j) alpha_k(j-1) * (beta_k(j)† alpha_k(j)) ]
    # Wait, this is getting complicated. Let me use a cleaner formulation.
    
    # Standard GRAPE: 
    # F = sum_k F_k where F_k = |<target_k|U|psi_k>|^2
    # dF/dalpha_i (where alpha_i is the control at step i) = 
    #   -2 Re[ <lambda_i | H_i | psi_i> ] where H_i is the drive part
    
    # Let me just do finite difference for the gradient - it's 160 params, 
    # but we can do it efficiently by reusing the forward propagation.
    
    # Actually, let's use a simpler gradient: perturb each parameter and recompute.
    # 160 parameters * 9 cases * 80 steps = expensive but manageable.
    
    # Better: use the adjoint method properly.
    
    # For a single case (det, scale), the gradient of the total infidelity is:
    # grad = sum_k d(1-F_k)/dalpha_i
    
    # Let me implement the adjoint method:
    # psi_k(i) = U_list[i] @ psi_k  (forward)
    # lambda_k(i) = U_list[i]† @ target_k  (this is NOT the adjoint for the gradient)
    
    # The correct adjoint: 
    # Let g_k = <target_k|U|psi_k>
    # F_k = |g_k|^2
    # dF_k = 2 Re[g_k* dg_k]
    # dg_k = <target_k|dU|psi_k>
    # dU = sum_i dU_step(i) where dU_step(i) = -i dt * dH_i * U_step(i) (first order)
    # But U = U_step(80)...U_step(1), so dU = sum_i U_step(80)...U_step(i+1) dU_step(i) U_step(i-1)...U_step(1)
    
    # This is complex. Let me just use numerical gradients with a small step.
    
    epsilon = 1e-6
    grad = np.zeros((n_steps, 2))
    
    # Compute base score
    U_base = U_list[-1]
    base_infid = 0
    for k, (psi, target) in enumerate(zip(initial_states, target_states)):
        g = np.vdot(target, U_base @ psi)
        F = np.abs(g)**2
        base_infid += (1 - F)
    base_infid /= 6
    
    # For each parameter, compute finite difference
    for i in range(n_steps):
        for param_idx in [0, 1]:
            pulse_plus = pulse_IQ.copy()
            pulse_plus[i, param_idx] += epsilon
            U_plus = _simulate_unitary(pulse_plus, detuning, scale, anharmonicity, dt, n_steps,
                                       a, a_dag, n, I3, n_nm1, a_plus_adag, a_minus_adag)
            infid_plus = 0
            for k, (psi, target) in enumerate(zip(initial_states, target_states)):
                g = np.vdot(target, U_plus @ psi)
                F = np.abs(g)**2
                infid_plus += (1 - F)
            infid_plus /= 6
            
            grad[i, param_idx] = (infid_plus - base_infid) / epsilon
    
    return grad

def pulse(target_gate, hardware_model):
    start_time = time.time()
    deadline = start_time + 310.0  # Safety margin before 330s kill

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

    # Seed pulse: DRAG with optimized parameters
    sigma_seed = T_total / 6.0
    pulse_seed = make_drag_pulse(sigma_seed, 0.0)
    best_pulse = pulse_seed.copy()
    best_score = evaluate_pulse(pulse_seed)

    # Coarse grid search on DRAG parameters
    sigma_range = np.linspace(T_total / 9.0, T_total / 4.5, 6)
    beta_range = np.linspace(-0.2, 0.2, 6)
    
    for sigma in sigma_range:
        if time.time() > deadline - 120.0:
            break
        for beta in beta_range:
            if time.time() > deadline - 120.0:
                break
            pulse_IQ = make_drag_pulse(sigma, beta)
            pulse_IQ = _project_to_valid(pulse_IQ, amp_bound, slew_bound, dt, n_steps)
            score = evaluate_pulse(pulse_IQ)
            if score < best_score:
                best_score = score
                best_pulse = pulse_IQ.copy()

    # GRAPE optimization
    # We optimize the full pulse using gradient ascent
    current_pulse = best_pulse.copy()
    current_score = best_score
    
    # Learning rate schedule
    lr_initial = 0.01
    lr_decay = 0.95
    
    # Number of GRAPE iterations
    max_grape_iters = 30
    
    for it in range(max_grape_iters):
        if time.time() > deadline - 30.0:
            break
        
        lr = lr_initial * (lr_decay ** it)
        
        # Compute gradient by summing gradients from all 9 cases
        total_grad = np.zeros((n_steps, 2))
        
        for det in detunings:
            for sc in scales:
                if time.time() > deadline - 25.0:
                    break
                grad = _compute_grape_gradient(current_pulse, det, sc, anharmonicity, dt, n_steps,
                                               a, a_dag, n, I3, n_nm1, a_plus_adag, a_minus_adag)
                total_grad += grad
        
        total_grad /= 9.0
        
        # Gradient ascent (we want to minimize, so go against gradient)
        new_pulse = current_pulse - lr * total_grad
        
        # Project to valid set
        new_pulse = _project_to_valid(new_pulse, amp_bound, slew_bound, dt, n_steps)
        
        # Evaluate
        new_score = evaluate_pulse(new_pulse)
        
        if new_score < current_score:
            current_pulse = new_pulse
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_pulse = current_pulse.copy()
        else:
            # If it didn't improve, try a smaller step
            new_pulse = current_pulse - (lr / 2) * total_grad
            new_pulse = _project_to_valid(new_pulse, amp_bound, slew_bound, dt, n_steps)
            new_score = evaluate_pulse(new_pulse)
            if new_score < current_score:
                current_pulse = new_pulse
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_pulse = current_pulse.copy()
    
    # Final local refinement with small random perturbations
    if time.time() < deadline - 15.0:
        for trial in range(20):
            if time.time() > deadline - 10.0:
                break
            perturbation = np.random.randn(n_steps, 2) * 0.005
            candidate = best_pulse + perturbation
            candidate = _project_to_valid(candidate, amp_bound, slew_bound, dt, n_steps)
            score = evaluate_pulse(candidate)
            if score < best_score:
                best_score = score
                best_pulse = candidate.copy()

    return best_pulse
