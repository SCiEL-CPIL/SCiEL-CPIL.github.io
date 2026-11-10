import numpy as np
import scipy.linalg as sla
import scipy.optimize as opt
import time
import math

def pulse(target_gate, hardware_model):
    """
    Returns an (80, 2) array of I and Q amplitudes.
    Strategy: Optimize DRAG pulse parameters (sigma, k) using a robust
    local optimizer (Nelder-Mead) with fast Liouvillian simulation.
    """
    start_time = time.time()
    deadline = start_time + 320.0  # 10s safety margin
    
    dt = 0.5
    n_steps = 80
    T = 40.0
    
    # Hardware parameters
    chi = -2 * math.pi * 0.25
    T1 = 25000.0
    Tphi = 40000.0
    
    # Operators
    a = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, math.sqrt(2), 0]
    ], dtype=complex)
    a_dag = a.conj().T
    n_op = a_dag @ a
    
    # Pre-compute static Liouvillian parts
    # H_static = detuning * n + 0.5 * chi * n(n-1)
    # Dissipators: a/sqrt(T1), n*sqrt(2/Tphi)
    
    def get_liouvillian(detuning, scale, I_t, Q_t):
        n_nm1 = n_op @ (n_op - np.eye(3))
        H_static = detuning * n_op + 0.5 * chi * n_nm1
        H_drive = 0.5 * scale * (I_t * (a + a_dag) + 1j * Q_t * (a - a_dag))
        H_total = H_static + H_drive
        
        # Lindblad Superoperators
        # L1 = a / sqrt(T1)
        L1 = a / math.sqrt(T1)
        L1_dag_L1 = L1.conj().T @ L1
        M_L1 = np.kron(L1.conj().T.T, L1) - 0.5 * np.kron(np.eye(3), L1_dag_L1) - 0.5 * np.kron(L1_dag_L1.T, np.eye(3))
        
        # L2 = n * sqrt(2/Tphi)
        L2 = n_op * math.sqrt(2.0 / Tphi)
        L2_dag_L2 = L2.conj().T @ L2
        M_L2 = np.kron(L2.conj().T.T, L2) - 0.5 * np.kron(np.eye(3), L2_dag_L2) - 0.5 * np.kron(L2_dag_L2.T, np.eye(3))
        
        M_diss = M_L1 + M_L2
        
        # Hamiltonian Superoperator
        M_H = -1j * (np.kron(np.eye(3), H_total) - np.kron(H_total.T, np.eye(3)))
        
        return M_H + M_diss

    def simulate_pulse(I, Q, detuning, scale):
        # Pre-compute Liouvillian for each step? No, H changes with I, Q.
        # But we can pre-compute M_diss.
        L1 = a / math.sqrt(T1)
        L1_dag_L1 = L1.conj().T @ L1
        M_L1 = np.kron(L1.conj().T.T, L1) - 0.5 * np.kron(np.eye(3), L1_dag_L1) - 0.5 * np.kron(L1_dag_L1.T, np.eye(3))
        
        L2 = n_op * math.sqrt(2.0 / Tphi)
        L2_dag_L2 = L2.conj().T @ L2
        M_L2 = np.kron(L2.conj().T.T, L2) - 0.5 * np.kron(np.eye(3), L2_dag_L2) - 0.5 * np.kron(L2_dag_L2.T, np.eye(3))
        M_diss = M_L1 + M_L2
        
        n_nm1 = n_op @ (n_op - np.eye(3))
        H_static_base = detuning * n_op + 0.5 * chi * n_nm1
        
        # Test states
        psi_0 = np.array([1, 0, 0], dtype=complex)
        psi_1 = np.array([0, 1, 0], dtype=complex)
        psi_p = (psi_0 + psi_1) / math.sqrt(2)
        psi_m = (psi_0 - psi_1) / math.sqrt(2)
        psi_ip = (psi_0 + 1j*psi_1) / math.sqrt(2)
        psi_im = (psi_0 - 1j*psi_1) / math.sqrt(2)
        
        test_states = [psi_0, psi_1, psi_p, psi_m, psi_ip, psi_im]
        targets = [psi_1, psi_0, (psi_1 + psi_0)/math.sqrt(2), (-psi_1 + psi_0)/math.sqrt(2), (1j*psi_1 + psi_0)/math.sqrt(2), (-1j*psi_1 + psi_0)/math.sqrt(2)]
        
        fidelities = []
        
        for psi, psi_t in zip(test_states, targets):
            rho = np.outer(psi, psi.conj())
            rho_vec = rho.flatten()
            
            for t_idx in range(n_steps):
                I_t = I[t_idx]
                Q_t = Q[t_idx]
                
                H_drive = 0.5 * scale * (I_t * (a + a_dag) + 1j * Q_t * (a - a_dag))
                H_total = H_static_base + H_drive
                
                M_H = -1j * (np.kron(np.eye(3), H_total) - np.kron(H_total.T, np.eye(3)))
                M_total = M_H + M_diss
                
                M_exp = sla.expm(M_total * dt)
                rho_vec = M_exp @ rho_vec
            
            rho_final = rho_vec.reshape(3, 3)
            rho_target = np.outer(psi_t, psi_t.conj())
            fidelity = np.real(np.trace(rho_target.conj().T @ rho_final))
            fidelity = max(0.0, min(1.0, fidelity))
            fidelities.append(fidelity)
        
        mean_fid = np.mean(fidelities)
        max_infid = 1.0 - np.min(fidelities)
        return mean_fid, max_infid

    def is_valid(I, Q):
        # Amplitude
        amps = np.sqrt(I**2 + Q**2)
        if np.max(amps) > 0.4:
            return False
        # Slew
        # p[0] from 0
        if math.sqrt(I[0]**2 + Q[0]**2) / dt > 0.12:
            return False
        # p[-1] to 0
        if math.sqrt(I[-1]**2 + Q[-1]**2) / dt > 0.12:
            return False
        # Internal
        dI = np.diff(I)
        dQ = np.diff(Q)
        slews = np.sqrt(dI**2 + dQ**2) / dt
        if np.max(slews) > 0.12:
            return False
        return True

    def parametric_pulse(sigma, k):
        t = np.arange(n_steps) * dt
        tc = T / 2.0
        gauss = np.exp(-(t - tc)**2 / (2 * sigma**2))
        area_sum = np.sum(gauss) * dt
        if area_sum == 0:
            return np.zeros(n_steps), np.zeros(n_steps)
        A = math.pi / area_sum
        I = A * gauss
        dI_dt = np.gradient(I, dt)
        Q = -k * dI_dt
        return I, Q

    def objective(params):
        sigma, k = params
        I, Q = parametric_pulse(sigma, k)
        
        if not is_valid(I, Q):
            return 1e6  # Penalty
        
        total_score = 0.0
        max_infid_all = 0.0
        
        detunings = [-2 * math.pi * 0.002, 0, 2 * math.pi * 0.002]
        scales = [0.97, 1.0, 1.03]
        
        count = 0
        for det in detunings:
            for sc in scales:
                mean_fid, max_infid = simulate_pulse(I, Q, det, sc)
                infid = 1.0 - mean_fid
                total_score += infid
                if max_infid > max_infid_all:
                    max_infid_all = max_infid
                count += 1
        
        mean_infid = total_score / count
        score = mean_infid + max_infid_all
        return score

    # Initial guess
    sigma0 = 40.0 / 6.0
    k0 = 0.0 # Start with no DRAG, let optimizer find it
    
    x0 = np.array([sigma0, k0])
    
    # Bounds for sigma and k
    # sigma must be large enough to satisfy slew and amplitude
    # k can be positive or negative
    bounds = [(3.0, 10.0), (-0.5, 0.5)]
    
    best_score = float('inf')
    best_params = x0
    best_I, best_Q = parametric_pulse(sigma0, k0)
    
    # Evaluate initial
    s = objective(x0)
    if s < best_score:
        best_score = s
        best_params = x0
        best_I, best_Q = parametric_pulse(sigma0, k0)
        
    # Optimize
    try:
        # Nelder-Mead is robust and doesn't need gradients
        result = opt.minimize(objective, x0, method='Nelder-Mead', 
                              options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-8})
        if result.fun < best_score:
            best_score = result.fun
            best_params = result.x
            best_I, best_Q = parametric_pulse(best_params[0], best_params[1])
    except Exception:
        pass
    
    # Check deadline
    if time.time() > deadline - 10:
        return np.column_stack((best_I, best_Q))
    
    # If time permits, try a finer search around the best point
    # Or just return the best found
    
    return np.column_stack((best_I, best_Q))
