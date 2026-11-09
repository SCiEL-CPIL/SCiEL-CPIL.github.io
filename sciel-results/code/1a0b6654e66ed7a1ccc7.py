"""Improved program for robust X-pi gate control pulse."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def pulse(target_gate, hardware_model):
    """
    Return an (80, 2) array of I and Q amplitudes (rad/ns) for a 40 ns X-pi gate.
    """
    # Time parameters
    dt = 0.5  # ns
    n_steps = 80
    t = np.arange(n_steps) * dt
    
    # Hardware parameters
    anharmonicity = -2 * np.pi * 0.25  # rad/ns
    amp_bound = 0.4
    slew_bound = 0.12
    detuning_bounds = (-2 * np.pi * 0.002, 2 * np.pi * 0.002)
    scale_bounds = (0.97, 1.03)
    T1 = 25000.0  # ns
    Tphi = 40000.0  # ns
    
    # DRAG pulse parameters
    sigma = 40.0 / 6.0  # ns
    t_center = 20.0  # ns
    
    # Initial DRAG pulse (Gaussian + derivative term)
    I = np.exp(-0.5 * ((t - t_center) / sigma) ** 2)
    # Normalize area to pi
    I_area = np.sum(I) * dt
    I = I * (np.pi / I_area)
    
    # Add DRAG term (derivative of Gaussian)
    dI = np.gradient(I, dt)
    Q = -beta * dI
    
    # Start with beta = 0 (pure Gaussian)
    beta = 0.0
    
    # Define the cost function
    def cost(params):
        beta_val = params[0]
        # Generate pulse
        I_p = I.copy()
        Q_p = -beta_val * dI
        
        # Check constraints
        # Amplitude bound
        amp = np.sqrt(I_p**2 + Q_p**2)
        if np.any(amp > amp_bound + 1e-12):
            return 1e10
        
        # Slew rate bound
        # From zero to step 1
        if np.sqrt(I_p[0]**2 + Q_p[0]**2) / dt > slew_bound + 1e-12:
            return 1e10
        # Between steps
        for i in range(n_steps - 1):
            di = I_p[i+1] - I_p[i]
            dq = Q_p[i+1] - Q_p[i]
            if np.sqrt(di**2 + dq**2) / dt > slew_bound + 1e-12:
                return 1e10
        # From last step to zero
        if np.sqrt(I_p[-1]**2 + Q_p[-1]**2) / dt > slew_bound + 1e-12:
            return 1e10
        
        # Compute infidelity over 9 cases
        detunings = [-2*np.pi*0.002, 0, 2*np.pi*0.002]
        scales = [0.97, 1.0, 1.03]
        
        fidelities = []
        for det in detunings:
            for scale in scales:
                # Build Hamiltonian parameters
                # a = [[0,1,0],[0,0,sqrt(2)],[0,0,0]]
                # n = a^dag a = diag(0,1,2)
                # H = det*n + 0.5*anharmonic*n*(n-1) + 0.5*scale*(I*(a+a^dag) + i*Q*(a-a^dag))
                
                # Compute infidelity for this case
                inf = _compute_infidelity(I_p, Q_p, det, scale, anharmonicity, T1, Tphi, dt, n_steps)
                fidelities.append(1 - inf)
        
        mean_fid = np.mean(fidelities)
        return 1 - mean_fid
    
    # Initial cost
    best_cost = cost([0.0])
    best_beta = 0.0
    
    # Optimize beta
    try:
        result = minimize(cost, [0.0], method='Nelder-Mead', 
                          options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10})
        if result.fun < best_cost:
            best_cost = result.fun
            best_beta = result.x[0]
    except Exception:
        pass
    
    # Generate final pulse
    I_final = I.copy()
    Q_final = -best_beta * dI
    
    # Check constraints and clip if necessary
    amp = np.sqrt(I_final**2 + Q_final**2)
    if np.any(amp > amp_bound):
        # Scale down
        scale_factor = amp_bound / np.max(amp)
        I_final *= scale_factor
        Q_final *= scale_factor
    
    # Verify slew rate
    # From zero to step 1
    if np.sqrt(I_final[0]**2 + Q_final[0]**2) / dt > slew_bound:
        # Reduce first step
        factor = slew_bound * dt / np.sqrt(I_final[0]**2 + Q_final[0]**2)
        I_final[0] *= factor
        Q_final[0] *= factor
    
    # Between steps
    for i in range(n_steps - 1):
        di = I_final[i+1] - I_final[i]
        dq = Q_final[i+1] - Q_final[i]
        dist = np.sqrt(di**2 + dq**2)
        if dist / dt > slew_bound:
            factor = slew_bound * dt / dist
            I_final[i+1] = I_final[i] + di * factor
            Q_final[i+1] = Q_final[i] + dq * factor
    
    # From last step to zero
    if np.sqrt(I_final[-1]**2 + Q_final[-1]**2) / dt > slew_bound:
        factor = slew_bound * dt / np.sqrt(I_final[-1]**2 + Q_final[-1]**2)
        I_final[-1] *= factor
        Q_final[-1] *= factor
    
    return np.column_stack([I_final, Q_final])


def _compute_infidelity(I, Q, detuning, scale, anharmonicity, T1, Tphi, dt, n_steps):
    """
    Compute mean infidelity to X gate over 5 initial states.
    """
    # Define operators
    # a = [[0,1,0],[0,0,sqrt(2)],[0,0,0]]
    a = np.zeros((3, 3), dtype=complex)
    a[0, 1] = 1
    a[1, 2] = np.sqrt(2)
    a_dag = a.conj().T
    
    # n = a^dag a
    n = a_dag @ a  # diag(0, 1, 2)
    
    # Target X gate on qubit subspace (|0>, |1>)
    # X = [[0,1],[1,0]] in qubit subspace
    # In 3-level system, X_pi maps |0> -> |1>, |1> -> |0>, |2> -> |2>
    X_target = np.zeros((3, 3), dtype=complex)
    X_target[0, 1] = 1
    X_target[1, 0] = 1
    X_target[2, 2] = 1  # |2> stays |2>
    
    # Initial states
    psi0 = np.array([1, 0, 0], dtype=complex)
    psi1 = np.array([0, 1, 0], dtype=complex)
    psi_plus = (psi0 + psi1) / np.sqrt(2)
    psi_minus = (psi0 - psi1) / np.sqrt(2)
    psi_plus_i = (psi0 + 1j*psi1) / np.sqrt(2)
    psi_minus_i = (psi0 - 1j*psi1) / np.sqrt(2)
    
    initial_states = [psi0, psi1, psi_plus, psi_minus, psi_plus_i, psi_minus_i]
    
    fidelities = []
    for psi_init in initial_states:
        # Evolve state
        psi_t = psi_init.copy()
        
        for step in range(n_steps):
            I_val = I[step]
            Q_val = Q[step]
            
            # Build Hamiltonian
            # H = detuning*n + 0.5*anharmonicity*n*(n-1) + 0.5*scale*(I*(a+a_dag) + i*Q*(a-a_dag))
            n_minus_1 = n - np.eye(3)
            H = (detuning * n + 
                 0.5 * anharmonicity * (n @ n_minus_1) +
                 0.5 * scale * (I_val * (a + a_dag) + 1j * Q_val * (a - a_dag)))
            
            # Build Liouvillian
            # L(rho) = -i[H, rho] + sum_k (L_k rho L_k^dag - 0.5{L_k^dag L_k, rho})
            # L_1 = a / sqrt(T1)
            # L_2 = n * sqrt(2/Tphi)
            
            L1 = a / np.sqrt(T1)
            L2 = n * np.sqrt(2.0 / Tphi)
            
            # Liouvillian superoperator
            # L(rho) = -i H rho + i rho H + L1 rho L1^dag + L2 rho L2^dag - 0.5(L1^dag L1 rho + rho L1^dag L1) - 0.5(L2^dag L2 rho + rho L2^dag L2)
            
            L1_dag_L1 = L1.conj().T @ L1
            L2_dag_L2 = L2.conj().T @ L2
            
            # Vectorize and apply
            def apply_liouvillian(rho):
                return (-1j * H @ rho + 1j * rho @ H +
                        L1 @ rho @ L1.conj().T + L2 @ rho @ L2.conj().T -
                        0.5 * (L1_dag_L1 @ rho + rho @ L1_dag_L1) -
                        0.5 * (L2_dag_L2 @ rho + rho @ L2_dag_L2))
            
            # Exponentiate: rho_{t+1} = exp(L * dt) rho_t
            # Use matrix exponential of the superoperator
            # Convert to superoperator form
            dim = 3
            super_op = np.zeros((dim*dim, dim*dim), dtype=complex)
            
            # For each basis element rho_ij, compute L(rho_ij)
            for i in range(dim):
                for j in range(dim):
                    rho_ij = np.zeros((dim, dim), dtype=complex)
                    rho_ij[i, j] = 1
                    result = apply_liouvillian(rho_ij)
                    # Vectorize result
                    result_vec = result.flatten()
                    # Column index for rho_ij
                    col_idx = i + j * dim
                    super_op[:, col_idx] = result_vec
            
            # Exponentiate
            from scipy.linalg import expm
            U_super = expm(super_op * dt)
            
            # Apply to current state
            rho_t = np.outer(psi_t, psi_t.conj())
            rho_vec = rho_t.flatten()
            rho_new_vec = U_super @ rho_vec
            rho_new = rho_new_vec.reshape((dim, dim))
            
            # Extract state (assuming pure state remains approximately pure)
            # Use eigenvalue decomposition
            eigvals, eigvecs = np.linalg.eigh(rho_new)
            # Take the largest eigenvalue
            max_idx = np.argmax(eigvals)
            psi_t = eigvecs[:, max_idx]
            # Ensure phase convention
            if np.abs(psi_t[0]) > 1e-10:
                phase = psi_t[0] / np.abs(psi_t[0])
                psi_t = psi_t / phase
        
        # Compute fidelity with target
        # F = |<psi_target|psi_final>|^2
        psi_target = X_target @ psi_init
        # Normalize
        psi_target = psi_target / np.linalg.norm(psi_target)
        psi_final = psi_t / np.linalg.norm(psi_t)
        
        fidelity = np.abs(np.vdot(psi_target, psi_final))**2
        fidelities.append(fidelity)
    
    return 1 - np.mean(fidelities)
# EVOLVE-BLOCK-END
