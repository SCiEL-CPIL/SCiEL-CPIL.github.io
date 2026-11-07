# EVOLVE-BLOCK-START
import numpy as np
from scipy.linalg import expm
import time

def pulse(target_gate, hardware_model):
    """
    Return an (80, 2) array of I and Q amplitudes (rad/ns) for a 40 ns X-pi gate.
    """
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
    
    # Build 3-level operators
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
    
    n_op_n_op_minus_I = n_op @ (n_op - I3)
    
    # Precompute kron forms
    kron_n_op = np.kron(I3, n_op)
    kron_n_op_T = np.kron(n_op.T, I3)
    kron_nnI = np.kron(I3, n_op_n_op_minus_I)
    kron_nnI_T = np.kron(n_op_n_op_minus_I.T, I3)
    kron_a_plus_ad = np.kron(I3, a + a_dag)
    kron_a_plus_ad_T = np.kron((a + a_dag).T, I3)
    kron_i_a_minus_ad = np.kron(I3, 1j * (a - a_dag))
    kron_i_a_minus_ad_T = np.kron((1j * (a - a_dag)).T, I3)
    kron_L1 = np.kron(L1, L1.T)
    kron_L1_dag_L1 = np.kron(I3, L1_dag_L1)
    kron_L1_dag_L1_T = np.kron(L1_dag_L1.T, I3)
    kron_L2 = np.kron(L2, L2.T)
    kron_L2_dag_L2 = np.kron(I3, L2_dag_L2)
    kron_L2_dag_L2_T = np.kron(L2_dag_L2.T, I3)
    
    # Dissipation part is constant
    L_diss = kron_L1 - 0.5 * (kron_L1_dag_L1 + kron_L1_dag_L1_T) \
           + kron_L2 - 0.5 * (kron_L2_dag_L2 + kron_L2_dag_L2_T)
    
    detunings = [-2 * np.pi * 0.002, 0.0, 2 * np.pi * 0.002]
    scales = [0.97, 1.0, 1.03]
    
    def build_L(detuning, scale, I_val, Q_val):
        L_comm_det = -1j * (detuning * (kron_n_op - kron_n_op_T) 
                       + 0.5 * anharmonicity * (kron_nnI - kron_nnI_T))
        
        L_comm_drive = -1j * (0.5 * scale * (I_val * (kron_a_plus_ad - kron_a_plus_ad_T)
                                        + Q_val * (kron_i_a_minus_ad - kron_i_a_minus_ad_T)))
        
        return L_comm_det + L_comm_drive + L_diss
    
    # Test states and targets
    states = [
        np.array([1, 0, 0], dtype=complex),
        np.array([0, 1, 0], dtype=complex),
        (np.array([1, 1, 0], dtype=complex) / np.sqrt(2)),
        (np.array([1, -1, 0], dtype=complex) / np.sqrt(2)),
        (np.array([1, 1j, 0], dtype=complex) / np.sqrt(2)),
        (np.array([1, -1j, 0], dtype=complex) / np.sqrt(2)),
    ]
    
    def get_target(state):
        target = np.zeros(3, dtype=complex)
        target[0] = -1j * state[1]
        target[1] = -1j * state[0]
        target[2] = state[2]
        return target
    
    targets = [get_target(s) for s in states]
    
    def simulate_and_score(pulse_IQ):
        all_infid = []
        
        for det in detunings:
            for scale in scales:
                rho = np.zeros((9,), dtype=complex)
                rho[0] = 1.0
                
                for i in range(N):
                    I_val = pulse_IQ[i, 0]
                    Q_val = pulse_IQ[i, 1]
                    L = build_L(det, scale, I_val, Q_val)
                    U = expm(dt * L)
                    rho = U @ rho
                
                rho_mat = rho.reshape(3, 3)
                
                case_infid = []
                for target in targets:
                    fid = np.real(np.conj(target) @ rho_mat @ target)
                    case_infid.append(1.0 - fid)
                all_infid.extend(case_infid)
        
        all_infid = np.array(all_infid)
        return np.mean(all_infid) + np.max(all_infid)
    
    def check_validity(pulse_IQ):
        amps = np.sqrt(pulse_IQ[:, 0]**2 + pulse_IQ[:, 1]**2)
        if np.any(amps > amplitude_bound + 1e-10):
            return False
        prev = np.zeros(2)
        for i in range(N):
            curr = pulse_IQ[i]
            if np.linalg.norm(curr - prev) / dt > slew_bound + 1e-10:
                return False
            prev = curr
        if np.linalg.norm(prev) / dt > slew_bound + 1e-10:
            return False
        return True
    
    def make_drag_pulse(alpha, sigma_factor, q_scale=1.0):
        """Create DRAG pulse with given parameters."""
        t = np.arange(N) * dt
        sigma = (total_time / 6.0) * sigma_factor
        center = total_time / 2.0
        gaussian = np.exp(-0.5 * ((t - center) / sigma)**2)
        
        # Offset so edges are zero
        edge_val = gaussian[0]
        gaussian = gaussian - edge_val
        
        # Normalize area to pi
        area = np.sum(gaussian) * dt
        if abs(area) > 1e-12:
            gaussian *= np.pi / area
        
        I = gaussian.copy()
        
        if abs(alpha) > 1e-12:
            dI_dt = np.gradient(I, dt)
            Q = alpha * q_scale * dI_dt
        else:
            Q = np.zeros(N)
        
        pulse_IQ = np.column_stack([I, Q])
        
        # Scale down if invalid
        amp_max = np.max(np.sqrt(I**2 + Q**2))
        if amp_max > amplitude_bound:
            pulse_IQ *= amplitude_bound / amp_max
        
        # Check slew
        prev = np.zeros(2)
        max_slew = 0.0
        for i in range(N):
            curr = pulse_IQ[i]
            max_slew = max(max_slew, np.linalg.norm(curr - prev) / dt)
            prev = curr
        max_slew = max(max_slew, np.linalg.norm(prev) / dt)
        
        if max_slew > slew_bound:
            pulse_IQ *= slew_bound / max_slew
        
        return pulse_IQ
    
    def make_shape_pulse(shape_params):
        """Create pulse from shape parameters: [amp, sigma_factor, alpha, q_scale, offset]"""
        amp = shape_params[0]
        sigma_factor = shape_params[1]
        alpha = shape_params[2]
        q_scale = shape_params[3]
        
        t = np.arange(N) * dt
        sigma = (total_time / 6.0) * sigma_factor
        center = total_time / 2.0
        gaussian = np.exp(-0.5 * ((t - center) / sigma)**2)
        
        # Offset so edges are zero
        edge_val = gaussian[0]
        gaussian = gaussian - edge_val
        
        # Normalize area to pi
        area = np.sum(gaussian) * dt
        if abs(area) > 1e-12:
            gaussian *= np.pi / area
        
        I = amp * gaussian
        Q = alpha * q_scale * np.gradient(I, dt)
        
        pulse_IQ = np.column_stack([I, Q])
        
        # Scale down if invalid
        amp_max = np.max(np.sqrt(I**2 + Q**2))
        if amp_max > amplitude_bound:
            pulse_IQ *= amplitude_bound / amp_max
        
        # Check slew
        prev = np.zeros(2)
        max_slew = 0.0
        for i in range(N):
            curr = pulse_IQ[i]
            max_slew = max(max_slew, np.linalg.norm(curr - prev) / dt)
            prev = curr
        max_slew = max(max_slew, np.linalg.norm(prev) / dt)
        
        if max_slew > slew_bound:
            pulse_IQ *= slew_bound / max_slew
        
        return pulse_IQ
    
    start_time = time.time()
    deadline = start_time + 310.0
    
    # Phase 1: Coarse grid search over DRAG parameters
    best_score = float('inf')
    best_pulse = make_drag_pulse(0.0, 1.0)
    
    # Wider range of alpha values for better DRAG correction
    alphas = np.array([-3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    sigma_factors = np.array([0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2])
    
    for alpha in alphas:
        for sf in sigma_factors:
            if time.time() > deadline - 80:
                break
            p = make_drag_pulse(alpha, sf)
            if check_validity(p):
                s = simulate_and_score(p)
                if s < best_score:
                    best_score = s
                    best_pulse = p.copy()
    
    # Phase 2: Local refinement with finer grid
    if time.time() < deadline - 40:
        # Find best alpha/sf from coarse grid
        best_alpha = 0.0
        best_sf = 1.0
        best_s = float('inf')
        for alpha in alphas:
            for sf in sigma_factors:
                p = make_drag_pulse(alpha, sf)
                if check_validity(p):
                    s = simulate_and_score(p)
                    if s < best_s:
                        best_s = s
                        best_alpha = alpha
                        best_sf = sf
        
        # Fine grid around best
        alpha_fine = np.linspace(best_alpha - 0.8, best_alpha + 0.8, 17)
        sf_fine = np.linspace(max(0.7, best_sf - 0.3), min(1.3, best_sf + 0.3), 13)
        
        for alpha in alpha_fine:
            for sf in sf_fine:
                if time.time() > deadline - 30:
                    break
                p = make_drag_pulse(alpha, sf)
                if check_validity(p):
                    s = simulate_and_score(p)
                    if s < best_score:
                        best_score = s
                        best_pulse = p.copy()
    
    # Phase 3: Gradient-based refinement using finite differences on shape parameters
    if time.time() < deadline - 20:
        # Use the best pulse as starting point and try small perturbations
        # in the shape parameter space
        best_shape = np.array([1.0, 1.0, 0.0, 1.0])  # [amp, sigma_factor, alpha, q_scale]
        
        # Try a few local perturbations
        perturbations = [
            [1.0, 1.0, -0.2, 1.0],
            [1.0, 1.0, 0.2, 1.0],
            [1.0, 0.95, 0.0, 1.0],
            [1.0, 1.05, 0.0, 1.0],
            [1.0, 1.0, 0.0, 0.8],
            [1.0, 1.0, 0.0, 1.2],
            [0.95, 1.0, 0.0, 1.0],
            [1.05, 1.0, 0.0, 1.0],
        ]
        
        for pert in perturbations:
            if time.time() > deadline - 15:
                break
            p = make_shape_pulse(pert)
            if check_validity(p):
                s = simulate_and_score(p)
                if s < best_score:
                    best_score = s
                    best_pulse = p.copy()
    
    return best_pulse
# EVOLVE-BLOCK-END
