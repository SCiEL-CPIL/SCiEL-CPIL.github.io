import numpy as np
from scipy.linalg import expm

def pulse(target_gate, hardware_model):
    # Hardware parameters
    chi = -2 * np.pi * 0.25
    amp_bound = 0.4
    slew_bound = 0.12
    det_bound = 2 * np.pi * 0.002
    T1 = 25000.0
    Tphi = 40000.0
    
    dt = 0.5
    N_steps = 80
    T_total = N_steps * dt
    
    # Basis operators
    a = np.zeros((3, 3), dtype=complex)
    a[1, 0] = 1.0
    a[2, 1] = np.sqrt(2.0)
    a_dag = a.conj().T
    n = a_dag @ a
    
    # Pre-compute Lindblad dissipator
    L1 = a / np.sqrt(T1)
    L2 = n * np.sqrt(2.0 / Tphi)
    
    def lindblad_term(L_op):
        L_op_dag = L_op.conj().T
        L_op_dag_L_op = L_op_dag @ L_op
        term1 = np.kron(L_op.conj(), L_op)
        term2 = -0.5 * np.kron(np.eye(3), L_op_dag_L_op)
        term3 = -0.5 * np.kron(L_op_dag_L_op, np.eye(3))
        return term1 + term2 + term3
    
    L_diss = lindblad_term(L1) + lindblad_term(L2)
    
    # Pre-compute Hamiltonian components
    H_anh = 0.5 * chi * (n @ n - n)  # diag(0, 0, chi)
    X_op = a + a_dag
    Y_op = 1j * (a - a_dag)
    
    def generate_pulse(sigma, beta):
        t = (np.arange(N_steps) + 0.5) * dt
        t0 = T_total / 2.0
        
        gauss = np.exp(-(t - t0)**2 / (2 * sigma**2))
        dgauss = -(t - t0) / (sigma**2) * gauss
        
        area = np.sum(gauss) * dt
        A = np.pi / area if area > 0 else 0
        
        I = A * gauss
        Q = -beta * A * dgauss
        
        # Enforce amplitude bound
        amp = np.sqrt(I**2 + Q**2)
        max_amp = np.max(amp)
        if max_amp > amp_bound:
            scale_factor = amp_bound / max_amp
            I *= scale_factor
            Q *= scale_factor
        
        # Enforce slew bound by iterative reduction if needed
        def check_slew(I, Q):
            dI = np.diff(I, prepend=0.0)
            dQ = np.diff(Q, prepend=0.0)
            dI = np.append(dI, -I[-1])
            dQ = np.append(dQ, -Q[-1])
            slew = np.sqrt(dI**2 + dQ**2) / dt
            return np.max(slew)
        
        if check_slew(I, Q) > slew_bound:
            # Reduce amplitude to meet slew constraint
            max_slew = check_slew(I, Q)
            slew_scale = slew_bound / max_slew
            I *= slew_scale
            Q *= slew_scale
        
        return np.column_stack((I, Q))

    def simulate_and_score(pulse_arr):
        I = pulse_arr[:, 0]
        Q = pulse_arr[:, 1]
        
        states = [
            np.array([1, 0, 0], dtype=complex),
            np.array([0, 1, 0], dtype=complex),
            np.array([1, 1, 0], dtype=complex) / np.sqrt(2),
            np.array([1, -1, 0], dtype=complex) / np.sqrt(2),
            np.array([1, 1j, 0], dtype=complex) / np.sqrt(2),
            np.array([1, -1j, 0], dtype=complex) / np.sqrt(2)
        ]
        
        X_mat = np.array([[0,1,0],[1,0,0],[0,0,1]], dtype=complex)
        targets = [X_mat @ psi for psi in states]
        
        dets = [-det_bound, 0.0, det_bound]
        scales = [0.97, 1.0, 1.03]
        
        infidelities = []
        
        for det in dets:
            for scale in scales:
                case_infids = []
                for k, psi_init in enumerate(states):
                    rho = np.outer(psi_init, psi_init.conj())
                    rho_vec = rho.flatten()
                    
                    for i in range(N_steps):
                        I_val = I[i]
                        Q_val = Q[i]
                        
                        H_drive = 0.5 * scale * (I_val * X_op + Q_val * Y_op)
                        H = det * n + H_anh + H_drive
                        
                        L_ham = -1j * (np.kron(np.eye(3), H) - np.kron(H.T, np.eye(3)))
                        L_total = L_ham + L_diss
                        
                        U = expm(L_total * dt)
                        rho_vec = U @ rho_vec
                    
                    rho_final = rho_vec.reshape((3, 3))
                    t_vec = targets[k]
                    F = np.real(np.vdot(t_vec, rho_final @ t_vec))
                    F = np.clip(F, 0.0, 1.0)
                    case_infids.append(1.0 - F)
                
                infidelities.append(np.mean(case_infids))
        
        return np.mean(infidelities) + np.max(infidelities)

    # Grid search with finer resolution
    best_score = float('inf')
    best_sigma = 6.0
    best_beta = 0.0
    
    sigmas = np.linspace(4.0, 8.0, 20)
    betas = np.linspace(-0.3, 0.3, 25)
    
    for sig in sigmas:
        for bet in betas:
            p = generate_pulse(sig, bet)
            
            # Validity checks
            amp = np.sqrt(p[:,0]**2 + p[:,1]**2)
            if np.max(amp) > amp_bound + 1e-9:
                continue
            
            dI = np.diff(p[:,0], prepend=0.0)
            dQ = np.diff(p[:,1], prepend=0.0)
            dI = np.append(dI, -p[-1,0])
            dQ = np.append(dQ, -p[-1,1])
            slew = np.max(np.sqrt(dI**2 + dQ**2) / dt)
            if slew > slew_bound + 1e-9:
                continue
            
            score = simulate_and_score(p)
            if score < best_score:
                best_score = score
                best_sigma = sig
                best_beta = bet

    final_pulse = generate_pulse(best_sigma, best_beta)
    
    # Final safety check
    amp = np.sqrt(final_pulse[:,0]**2 + final_pulse[:,1]**2)
    if np.max(amp) > amp_bound:
        final_pulse *= amp_bound / np.max(amp)
    
    return final_pulse

pulse = pulse
