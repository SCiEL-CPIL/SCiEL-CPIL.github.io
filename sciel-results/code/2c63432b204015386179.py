import numpy as np
import scipy.linalg as sla
import time
import math

def pulse(target_gate, hardware_model):
    """
    Returns an (80, 2) array of I and Q amplitudes.
    Strategy: Analytic DRAG design + fast local search.
    1. Construct DRAG pulse: I(t) ~ Gaussian, Q(t) ~ -k * dI/dt.
    2. Optimize k (DRAG strength) and global scale to minimize infidelity.
    3. Use pre-computed static Liouvillian parts for speed.
    """
    start_time = time.time()
    deadline = start_time + 320.0 # 10s safety margin

    dt = 0.5
    n_steps = 80
    T = 40.0

    # Hardware constants
    chi = -2 * math.pi * 0.25  # Anharmonicity (rad/ns)
    T1 = 25000.0
    Tphi = 40000.0
    
    # Nominal params for optimization (we optimize for nominal, hope it's robust)
    # Or optimize for worst case? The prompt says "robust control pulses".
    # The score is mean + max over 9 corners.
    # Optimizing for nominal is fast. Optimizing for worst-case is safer but slower.
    # Let's optimize for a weighted average or just nominal, as DRAG is known to be robust.
    
    # Operators
    a = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, math.sqrt(2), 0]
    ], dtype=complex)
    a_dag = a.conj().T
    n_op = a_dag @ a
    n_nm1 = n_op @ (n_op - np.eye(3))
    
    # Static Hamiltonian part (detuning=0 for nominal)
    H_static_nominal = 0.5 * chi * n_nm1
    
    # Dissipators
    L1 = a / math.sqrt(T1)
    L2 = n_op * math.sqrt(2.0 / Tphi)
    
    # Pre-compute Lindblad superoperator for dissipators (static)
    def lindblad_superop(L):
        LdagL = L.conj().T @ L
        LdagT = L.conj().T.T
        M = np.kron(LdagT, L) - 0.5 * np.kron(np.eye(3), LdagL) - 0.5 * np.kron(LdagL.T, np.eye(3))
        return M
        
    M_diss = lindblad_superop(L1) + lindblad_superop(L2)
    
    # Pre-compute basis for drive Hamiltonian
    # H_drive = 0.5 * scale * (I * (a+adag) + i Q * (a-adag))
    # Let H_I = 0.5 * (a + a_dag)
    # Let H_Q = 0.5 * i * (a - a_dag)
    H_I_base = 0.5 * (a + a_dag)
    H_Q_base = 0.5 * 1j * (a - a_dag)
    
    # Superoperators for H_I and H_Q (static structure, coefficients I, Q vary)
    # M_H = -i (I ⊗ H - H^T ⊗ I)
    # M_H_drive = -i (I ⊗ (I_t H_I_base + Q_t H_Q_base) - (I_t H_I_base + Q_t H_Q_base)^T ⊗ I)
    # = I_t * M_I_base + Q_t * M_Q_base
    M_I_base = -1j * (np.kron(np.eye(3), H_I_base) - np.kron(H_I_base.T, np.eye(3)))
    M_Q_base = -1j * (np.kron(np.eye(3), H_Q_base) - np.kron(H_Q_base.T, np.eye(3)))
    
    # Test states
    psi_0 = np.array([1, 0, 0], dtype=complex)
    psi_1 = np.array([0, 1, 0], dtype=complex)
    psi_p = (psi_0 + psi_1) / math.sqrt(2)
    psi_m = (psi_0 - psi_1) / math.sqrt(2)
    psi_ip = (psi_0 + 1j*psi_1) / math.sqrt(2)
    psi_im = (psi_0 - 1j*psi_1) / math.sqrt(2)
    
    test_states = [psi_0, psi_1, psi_p, psi_m, psi_ip, psi_im]
    target_states = [psi_1, psi_0, (psi_1 + psi_0)/math.sqrt(2), (-psi_1 + psi_0)/math.sqrt(2), 
                     (1j*psi_1 + psi_0)/math.sqrt(2), (-1j*psi_1 + psi_0)/math.sqrt(2)]
    
    target_rhos = [np.outer(psi, psi.conj()) for psi in target_states]
    init_rhos = [np.outer(psi, psi.conj()).flatten() for psi in test_states]

    def simulate_fast(I, Q, detuning, scale):
        """
        Fast simulation using pre-computed superoperators.
        Returns mean infidelity and max infidelity.
        """
        # H_static = detuning * n + 0.5 * chi * n(n-1)
        H_static = detuning * n_op + 0.5 * chi * n_nm1
        M_H_static = -1j * (np.kron(np.eye(3), H_static) - np.kron(H_static.T, np.eye(3)))
        
        # Total static Liouvillian
        M_static = M_H_static + M_diss
        
        # Pre-compute exp(M_static * dt)
        M_static_exp = sla.expm(M_static * dt)
        
        infidelities = []
        
        # We simulate all 6 states in parallel? 
        # Or just loop. 6 states * 80 steps * expm(9x9) is the bottleneck.
        # Can we batch? 
        # vec(rho) is 9x1. We can stack 6 states into a 9x6 matrix.
        # M @ X works.
        
        Rho_stack = np.column_stack(init_rhos)
        
        for t_idx in range(n_steps):
            I_t = I[t_idx]
            Q_t = Q[t_idx]
            
            # M_total = M_static + I_t * M_I_base * scale + Q_t * M_Q_base * scale
            # Note: scale is a scalar factor for the drive
            M_drive = scale * (I_t * M_I_base + Q_t * M_Q_base)
            M_total = M_static + M_drive
            
            M_total_exp = sla.expm(M_total * dt)
            Rho_stack = M_total_exp @ Rho_stack
            
        # Calculate fidelities
        for i in range(6):
            rho_final = Rho_stack[:, i].reshape(3, 3)
            rho_target = target_rhos[i]
            fid = np.real(np.trace(rho_target.conj().T @ rho_final))
            fid = max(0.0, min(1.0, fid))
            infid = 1.0 - fid
            infidelities.append(infid)
            
        mean_infid = np.mean(infidelities)
        max_infid = np.max(infidelities)
        return mean_infid, max_infid

    def get_pulse_params(k_drag, area_scale):
        """
        Generate I, Q from DRAG parameters.
        """
        t = np.arange(n_steps) * dt
        tc = T / 2.0
        sigma = 40.0 / 6.0
        
        gauss = np.exp(-(t - tc)**2 / (2 * sigma**2))
        
        # Normalize area to pi * area_scale
        area_gauss = np.sum(gauss) * dt
        if area_gauss == 0:
            return np.zeros(n_steps), np.zeros(n_steps)
            
        A = (math.pi * area_scale) / area_gauss
        I = A * gauss
        
        dI_dt = np.gradient(I, dt)
        Q = -k_drag * dI_dt
        
        return I, Q

    def project_pulse(I, Q):
        """Project to constraints."""
        I = I.copy()
        Q = Q.copy()
        
        # Amplitude constraint
        amps = np.sqrt(I**2 + Q**2)
        max_amp = np.max(amps)
        if max_amp > 0.4:
            scale = 0.4 / max_amp
            I *= scale
            Q *= scale
            
        # Slew constraint
        # Check boundaries and steps
        max_slew = 0.0
        # Boundary 0 -> 1
        slew0 = math.sqrt(I[0]**2 + Q[0]**2) / dt
        max_slew = max(max_slew, slew0)
        # Boundary N -> 0
        slewe = math.sqrt(I[-1]**2 + Q[-1]**2) / dt
        max_slew = max(max_slew, slewe)
        
        for i in range(n_steps - 1):
            dI = I[i+1] - I[i]
            dQ = Q[i+1] - Q[i]
            dist = math.sqrt(dI**2 + dQ**2)
            slew = dist / dt
            if slew > max_slew:
                max_slew = slew
                
        if max_slew > 0.12:
            scale = 0.12 / max_slew
            I *= scale
            Q *= scale
            # Re-check amplitude
            amps = np.sqrt(I**2 + Q**2)
            max_amp = np.max(amps)
            if max_amp > 0.4:
                scale2 = 0.4 / max_amp
                I *= scale2
                Q *= scale2
                
        return I, Q

    def objective(params):
        """
        Objective: mean + max infidelity across 9 corners.
        """
        k_drag, area_scale = params
        
        if time.time() > deadline - 5.0:
            return 1e6 # Timeout penalty
            
        I, Q = get_pulse_params(k_drag, area_scale)
        I, Q = project_pulse(I, Q)
        
        total_score = 0.0
        max_infid_all = 0.0
        
        detunings = [-2 * math.pi * 0.002, 0, 2 * math.pi * 0.002]
        scales = [0.97, 1.0, 1.03]
        
        count = 0
        for det in detunings:
            for sc in scales:
                mean_infid, max_infid = simulate_fast(I, Q, det, sc)
                total_score += mean_infid
                if max_infid > max_infid_all:
                    max_infid_all = max_infid
                count += 1
                
        mean_infid = total_score / count
        score = mean_infid + max_infid_all
        return score

    # Initial guess
    # Standard DRAG k is related to chi.
    # Q = - (chi / 2) * dI/dt / (2*pi*Omega)? 
    # In our units, let's just search k.
    # Typical k for DRAG is small.
    k_init = 0.1
    area_init = 1.0
    
    # Grid search / Local optimization
    # Since the function is smooth and 2D, we can use a simple grid + refinement.
    
    best_score = float('inf')
    best_params = np.array([k_init, area_init])
    best_I, best_Q = get_pulse_params(k_init, area_init)
    best_I, best_Q = project_pulse(best_I, best_Q)
    
    # Coarse Grid
    k_range = np.linspace(-0.5, 0.5, 11)
    area_range = np.linspace(0.8, 1.2, 9)
    
    for k in k_range:
        for area in area_range:
            if time.time() > deadline - 30.0:
                break
            score = objective(np.array([k, area]))
            if score < best_score:
                best_score = score
                best_params = np.array([k, area])
                best_I, best_Q = get_pulse_params(k, area)
                best_I, best_Q = project_pulse(best_I, best_Q)
        if time.time() > deadline - 30.0:
            break
            
    # Refine around best
    if time.time() < deadline - 30.0:
        k_best, area_best = best_params
        k_refine = np.linspace(k_best - 0.05, k_best + 0.05, 11)
        area_refine = np.linspace(area_best - 0.05, area_best + 0.05, 11)
        
        for k in k_refine:
            for area in area_refine:
                if time.time() > deadline - 10.0:
                    break
                score = objective(np.array([k, area]))
                if score < best_score:
                    best_score = score
                    best_params = np.array([k, area])
                    best_I, best_Q = get_pulse_params(k, area)
                    best_I, best_Q = project_pulse(best_I, best_Q)
            if time.time() > deadline - 10.0:
                break

    # Final check
    I_final, Q_final = project_pulse(best_I, best_Q)
    
    # One last safety check: if the pulse is invalid, return seed-like pulse
    # (Though project_pulse should ensure validity)
    return np.column_stack((I_final, Q_final))
