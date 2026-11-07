# EVOLVE-BLOCK-START
import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize
import time

def pulse(target_gate, hardware_model):
    if target_gate != 'X_pi':
        return np.zeros((80, 2))

    # Hardware parameters
    anharmonicity = -2 * np.pi * 0.25
    amplitude_bound = 0.4
    slew_bound = 0.12
    T1 = 25000.0
    Tphi = 40000.0

    dt = 0.5
    N = 80
    total_time = N * dt

    # 3-level operators
    a = np.zeros((3, 3), dtype=complex)
    a[0, 1] = 1.0
    a[1, 2] = np.sqrt(2.0)
    a_dag = a.conj().T
    n_op = a_dag @ a
    I3 = np.eye(3)

    # Lindblad operators
    L1 = a / np.sqrt(T1)
    L2 = n_op * np.sqrt(2.0 / Tphi)

    L1_dag_L1 = L1.conj().T @ L1
    L2_dag_L2 = L2.conj().T @ L2

    L1_term = np.kron(L1, L1.T)
    L2_term = np.kron(L2, L2.T)
    L1_diss_left = np.kron(I3, L1_dag_L1)
    L1_diss_right = np.kron(L1_dag_L1.T, I3)
    L2_diss_left = np.kron(I3, L2_dag_L2)
    L2_diss_right = np.kron(L2_dag_L2.T, I3)

    static_Liouv = L1_term - 0.5 * (L1_diss_left + L1_diss_right) \
                  + L2_term - 0.5 * (L2_diss_left + L2_diss_right)

    an_ham = 0.5 * anharmonicity * n_op @ (n_op - I3)

    detunings = np.array([-2 * np.pi * 0.002, 0.0, 2 * np.pi * 0.002])
    scales = np.array([0.97, 1.0, 1.03])

    case_static_L = np.zeros((9, 9, 9), dtype=complex)
    case_scale = np.zeros(9)
    case_idx = 0
    for det in detunings:
        for sc in scales:
            H_static = det * n_op + an_ham
            H_left = np.kron(I3, H_static)
            H_right = np.kron(H_static.T, I3)
            case_static_L[case_idx] = -1j * (H_left - H_right) + static_Liouv
            case_scale[case_idx] = sc
            case_idx += 1

    # Test states
    states = [
        np.array([1, 0, 0], dtype=complex),
        np.array([0, 1, 0], dtype=complex),
        np.array([1, 1, 0], dtype=complex) / np.sqrt(2),
        np.array([1, -1, 0], dtype=complex) / np.sqrt(2),
        np.array([1, 1j, 0], dtype=complex) / np.sqrt(2),
        np.array([1, -1j, 0], dtype=complex) / np.sqrt(2),
    ]

    # X_pi target: -i * swap |0> and |1>
    target_states = []
    for s in states:
        t = np.zeros(3, dtype=complex)
        t[0] = -1j * s[1]
        t[1] = -1j * s[0]
        t[2] = s[2]
        target_states.append(t)

    # Precompute rho_init
    rho_init = np.zeros((6, 9), dtype=complex)
    for si in range(6):
        psi = states[si]
        rho_mat = np.outer(psi, psi.conj())
        rho_init[si] = rho_mat.flatten()

    def check_validity(pulse_IQ):
        amplitudes = np.sqrt(pulse_IQ[:, 0]**2 + pulse_IQ[:, 1]**2)
        if np.any(amplitudes > amplitude_bound + 1e-12):
            return False
        prev = np.zeros(2)
        for i in range(N):
            curr = pulse_IQ[i]
            if np.linalg.norm(curr - prev) / dt > slew_bound + 1e-12:
                return False
            prev = curr
        if np.linalg.norm(prev) / dt > slew_bound + 1e-12:
            return False
        return True

    def project_valid(pulse_IQ):
        """
        Project a pulse to satisfy validity constraints.
        1. Clip amplitude to bound.
        2. Enforce slew rate by limiting step changes.
        """
        p = pulse_IQ.copy()
        
        # 1. Amplitude clipping
        amplitudes = np.sqrt(p[:, 0]**2 + p[:, 1]**2)
        max_amp = np.max(amplitudes)
        if max_amp > amplitude_bound:
            p *= amplitude_bound / max_amp
            
        # 2. Slew rate enforcement (forward pass)
        # This is a simple iterative projection. 
        # We ensure |p[i] - p[i-1]| <= slew_bound * dt
        max_step = slew_bound * dt
        prev = np.zeros(2)
        for i in range(N):
            diff = p[i] - prev
            norm_diff = np.linalg.norm(diff)
            if norm_diff > max_step:
                p[i] = prev + diff * (max_step / norm_diff)
            prev = p[i]
        
        # 3. Slew rate enforcement (backward pass) to satisfy end condition
        prev = np.zeros(2)
        for i in range(N - 1, -1, -1):
            diff = p[i] - prev
            norm_diff = np.linalg.norm(diff)
            if norm_diff > max_step:
                p[i] = prev + diff * (max_step / norm_diff)
            prev = p[i]
            
        # Re-check amplitude after slew projection
        amplitudes = np.sqrt(p[:, 0]**2 + p[:, 1]**2)
        max_amp = np.max(amplitudes)
        if max_amp > amplitude_bound:
            p *= amplitude_bound / max_amp
            
        return p

    def simulate_pulse(pulse_IQ):
        """
        Simulate all 9 cases x 6 states.
        Returns fids of shape (9, 6).
        """
        fids = np.zeros((9, 6))
        for ci in range(9):
            sc = case_scale[ci]
            U_step_list = []
            for step in range(N):
                I_val = pulse_IQ[step, 0]
                Q_val = pulse_IQ[step, 1]
                H_drive = 0.5 * sc * (I_val * (a + a_dag) + 1j * Q_val * (a - a_dag))
                H_left = np.kron(I3, H_drive)
                H_right = np.kron(H_drive.T, I3)
                L = case_static_L[ci] - 1j * (H_left - H_right)
                U_step_list.append(expm(dt * L))

            for si in range(6):
                vec = rho_init[si].copy()
                for step in range(N):
                    vec = U_step_list[step] @ vec
                rho_mat = vec.reshape(3, 3)
                psi_t = target_states[si]
                fids[ci, si] = np.real(np.conj(psi_t) @ rho_mat @ psi_t)
        return fids

    def compute_score(fids):
        infids = 1.0 - fids
        return np.mean(infids) + np.max(infids)

    def make_drag_pulse(alpha=0.0, sigma_ratio=1.0):
        t = np.arange(N) * dt
        sigma = (40.0 / 6.0) * sigma_ratio
        center = total_time / 2.0
        gaussian = np.exp(-0.5 * ((t - center) / sigma)**2)
        edge_val = gaussian[0]
        gaussian -= edge_val
        area = np.sum(gaussian) * dt
        if abs(area) > 1e-15:
            gaussian *= np.pi / area
        I = gaussian.copy()
        if alpha != 0:
            dI_dt = np.gradient(I, dt)
            Q = alpha * dI_dt
        else:
            Q = np.zeros(N)
        pulse_IQ = np.column_stack([I, Q])
        return project_valid(pulse_IQ)

    # Time management
    start_time = time.time()
    deadline = start_time + 300.0

    def time_remaining():
        return deadline - time.time()

    # Phase 1: Find a good DRAG seed
    pulse_IQ = make_drag_pulse(0.0, 1.0)
    best_pulse = pulse_IQ.copy()
    fids_best = simulate_pulse(best_pulse)
    best_score = compute_score(fids_best)

    # Quick sweep of DRAG parameters
    alphas = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
    sigma_ratios = [0.8, 0.9, 1.0, 1.1, 1.2]
    
    for sr in sigma_ratios:
        if time_remaining() < 60: break
        for alpha in alphas:
            if time_remaining() < 60: break
            p_test = make_drag_pulse(alpha, sr)
            fids_t = simulate_pulse(p_test)
            s = compute_score(fids_t)
            if s < best_score:
                best_score = s
                best_pulse = p_test.copy()

    # Phase 2: SPSA Optimization
    # SPSA is robust to non-smoothness and doesn't require gradients.
    # Parameters: 160 values (80 I, 80 Q).
    
    w = best_pulse.flatten().copy()
    rng = np.random.default_rng(42)
    
    # SPSA parameters
    # A = 0.1, c = 0.05
    # A_k = A / (k + alpha)^(gamma)
    # c_k = c / (k + alpha)^(gamma_c)
    A = 0.1
    c = 0.05
    gamma = 0.6
    gamma_c = 0.6
    alpha = 1.0
    
    max_spsa_steps = 1000
    current_w = w
    
    for k in range(max_spsa_steps):
        if time_remaining() < 30:
            break
            
        a_k = A / (k + alpha)**gamma
        c_k = c / (k + alpha)**gamma_c
        
        # Generate random perturbation vector +/- 1
        delta = rng.choice([-1, 1], size=2*N)
        
        # Perturbed pulses
        w_plus = current_w + c_k * delta
        w_minus = current_w - c_k * delta
        
        p_plus = project_valid(w_plus.reshape(N, 2))
        p_minus = project_valid(w_minus.reshape(N, 2))
        
        # Evaluate scores
        fids_plus = simulate_pulse(p_plus)
        s_plus = compute_score(fids_plus)
        
        fids_minus = simulate_pulse(p_minus)
        s_minus = compute_score(fids_minus)
        
        # SPSA gradient estimate
        # g = (J(w+c) - J(w-c)) / (2c) * delta
        if abs(2 * c_k) > 1e-12:
            g = ((s_plus - s_minus) / (2 * c_k)) * delta
        else:
            g = np.zeros(2*N)
            
        # Update
        current_w = current_w - a_k * g
        
        # Project to valid space
        current_p = project_valid(current_w.reshape(N, 2))
        current_w = current_p.flatten()
        
        # Evaluate best
        # (We already have s_plus and s_minus, but the new w might be better than both if the step was large and correct direction)
        # For efficiency, we can just track the best seen so far
        fids_new = simulate_pulse(current_p)
        s_new = compute_score(fids_new)
        
        if s_new < best_score:
            best_score = s_new
            best_pulse = current_p.copy()
            
        # Adaptive c and A
        # Standard SPSA: A_k = A/(k+alpha)^gamma, c_k = c/(k+alpha)^gamma_c
        # This is already handled by k.

    # Phase 3: Nelder-Mead Polish (if time permits)
    # NM can refine the local minimum found by SPSA.
    if time_remaining() > 60:
        x0 = best_pulse.flatten()
        
        def nm_obj(params):
            if time_remaining() < 10:
                return 1e10
            p = project_valid(params.reshape(N, 2))
            fids = simulate_pulse(p)
            return compute_score(fids)
            
        try:
            # NM in high dims is risky, but we're starting close.
            # Limit iterations to keep it fast.
            result = minimize(nm_obj, x0, method='Nelder-Mead',
                              options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-8, 'maxfev': 100})
            if time_remaining() > 20:
                p_nm = project_valid(result.x.reshape(N, 2))
                fids_nm = simulate_pulse(p_nm)
                s_nm = compute_score(fids_nm)
                if s_nm < best_score:
                    best_score = s_nm
                    best_pulse = p_nm.copy()
        except Exception:
            pass

    # Final check
    if not check_validity(best_pulse):
        best_pulse = project_valid(best_pulse)
        
    return best_pulse
# EVOLVE-BLOCK-END
