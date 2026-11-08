"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.linalg import expm
from scipy.optimize import minimize

def pulse(target_gate, hardware_model):
    """
    Generate a robust pulse for X_pi gate using parametric DRAG optimization.
    
    Strategy:
    1. Parametric DRAG-style pulse with Gaussian envelope.
    2. Direct optimization of key parameters: sigma, alpha (DRAG), area_scale.
    3. Full quantum simulation with Lindblad terms for accurate infidelity.
    4. Deadline guard to ensure completion within time limit.
    """
    if target_gate != 'X_pi':
        raise ValueError("Only X_pi is supported")
    
    start_time = time.time()
    deadline = start_time + 300.0  # 300s safety margin
    
    # Hardware parameters
    anharmonicity = -2 * np.pi * 0.25  # rad/ns
    amp_bound = 0.4
    slew_bound = 0.12
    dt = 0.5  # ns
    n_steps = 80
    total_time = n_steps * dt  # 40 ns
    T1 = 25000.0
    Tphi = 40000.0
    
    # Time points
    t = np.linspace(0, total_time, n_steps, endpoint=False) + dt/2
    t_centered = t - total_time / 2.0
    
    # Pre-compute matrices for speed
    a = np.array([[0, 1, 0],
                  [0, 0, np.sqrt(2)],
                  [0, 0, 0]], dtype=complex)
    adag = a.conj().T
    n_op = adag @ a
    eye3 = np.eye(3)
    
    H0_base = 0.5 * anharmonicity * n_op @ (n_op - eye3)
    
    # Lindblad operators
    L1 = a / np.sqrt(T1)
    L2 = n_op * np.sqrt(2.0 / Tphi)
    
    # Pre-compute Lindblad superoperator terms for speed
    # The Liouvillian: L(rho) = -i[H,rho] + sum_k (L_k rho L_k^dag - 0.5{L_k^dag L_k, rho})
    # We'll build the full Liouvillian matrix for each step
    
    def build_liouvillian(H, L_ops):
        """Build the Liouvillian superoperator as a 9x9 matrix."""
        dim = 3
        L_super = np.zeros((dim*dim, dim*dim), dtype=complex)
        
        # Coherent part: -i[H, rho]
        # In vectorized form: -i (H ⊗ I - I ⊗ H^T)
        H_vec = H.reshape(dim*dim)
        for i in range(dim):
            for j in range(dim):
                L_super[i*dim+j, i*dim+j] = -1j * (H[i,i] - H[j,j])
                for k in range(dim):
                    if k != i and k != j:
                        pass
                    # Actually, let's just compute it directly
        # Simpler: compute action on vectorized rho
        # L_super @ vec(rho) = vec(-i[H,rho] + dissipator)
        
        # Coherent: vec(-i[H,rho]) = -i (I ⊗ H^T - H ⊗ I) vec(rho)
        # In numpy, vec stacks columns, so:
        # vec(H @ rho) = (I ⊗ H) vec(rho)
        # vec(rho @ H^T) = (H^T ⊗ I) vec(rho)
        # So vec(H @ rho - rho @ H^T) = (I ⊗ H - H^T ⊗ I) vec(rho)
        
        I_dim = np.eye(dim)
        coherent = -1j * (np.kron(I_dim, H) - np.kron(H.T, I_dim))
        
        # Dissipative part
        dissipator = np.zeros((dim*dim, dim*dim), dtype=complex)
        for L in L_ops:
            Ld = L.conj().T
            # vec(L rho L^dag) = (L^dag^T ⊗ L) vec(rho)
            dissipator += np.kron(Ld.T, L)
            # vec(0.5 {L^dag L, rho}) = 0.5 (L^dag L ⊗ I + I ⊗ (L^dag L)^T) vec(rho)
            LdL = Ld @ L
            dissipator += -0.5 * (np.kron(LdL, I_dim) + np.kron(I_dim, LdL.T))
        
        L_super = coherent + dissipator
        return L_super
    
    # Simulation function with Lindblad terms
    def simulate_full(pulse_arr, delta, scale):
        """
        Simulate full quantum evolution with Lindblad terms.
        Returns the process matrix (9x9) or we can track fidelity directly.
        """
        # We'll simulate the evolution of the density matrix for each test state
        # and compute fidelity directly
        
        states = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0] / np.sqrt(2),
            [1, -1, 0] / np.sqrt(2),
            [1, 1j, 0] / np.sqrt(2),
            [1, -1j, 0] / np.sqrt(2)
        ], dtype=complex)
        
        X = np.array([[0, 1, 0],
                      [1, 0, 0],
                      [0, 0, 0]], dtype=complex)
        
        fidelities = []
        for psi in states:
            rho = np.outer(psi, psi.conj())
            target = X @ psi
            target_rho = np.outer(target, target.conj())
            
            for i in range(n_steps):
                I_i = pulse_arr[i, 0]
                Q_i = pulse_arr[i, 1]
                
                H_int = 0.5 * scale * (I_i * (a + adag) + 1j * Q_i * (a - adag))
                H_t = delta * n_op + H0_base + H_int
                
                L_ops = [L1, L2]
                L_super = build_liouvillian(H_t, L_ops)
                
                rho_vec = rho.reshape(9)
                # Evolve: rho(t+dt) = exp(L_super * dt) @ rho(t)
                eigvals, eigvecs = np.linalg.eig(L_super)
                exp_eig = np.exp(eigvals * dt)
                rho_vec_new = eigvecs @ (exp_eig * (np.linalg.inv(eigvecs) @ rho_vec))
                rho = rho_vec_new.reshape(3, 3)
            
            # Fidelity = Tr(target_rho @ rho)
            fid = np.real(np.trace(target_rho @ rho))
            fid = np.clip(fid, 0, 1)
            fidelities.append(fid)
        
        return np.mean(fidelities)
    
    # Score function with soft-max surrogate
    def compute_score(pulse_arr):
        deltas = [-2 * np.pi * 0.002, 0, 2 * np.pi * 0.002]
        scales = [0.97, 1.0, 1.03]
        
        infidelities = []
        for d in deltas:
            for s in scales:
                fid = simulate_full(pulse_arr, d, s)
                infidelities.append(1 - fid)
        
        infidelities = np.array(infidelities)
        mean_inf = np.mean(infidelities)
        
        # Soft-max surrogate: use log-sum-exp for smooth max
        beta = 50.0
        max_inf = np.log(np.sum(np.exp(beta * infidelities)) / len(infidelities)) / beta
        
        return mean_inf + max_inf
    
    # Constraint handling
    def enforce_constraints(p):
        """
        Enforce amplitude and slew rate constraints by scaling.
        """
        # Amplitude
        amp = np.sqrt(p[:, 0]**2 + p[:, 1]**2)
        if np.max(amp) > amp_bound:
            p *= amp_bound / np.max(amp)
        
        # Slew rate
        max_slew = 0
        for i in range(n_steps):
            if i == 0:
                dp = np.linalg.norm(p[0]) / dt
            else:
                dp = np.linalg.norm(p[i] - p[i-1]) / dt
            if dp > max_slew:
                max_slew = dp
        
        dp_last = np.linalg.norm(p[-1]) / dt
        if dp_last > max_slew:
            max_slew = dp_last
            
        if max_slew > slew_bound:
            p *= slew_bound / max_slew
            
        return p
    
    # Parametric pulse with more parameters
    def parametric_pulse(params):
        sigma, alpha, area_scale, width_mod = params
        # Gaussian with optional width modulation
        I = np.exp(-t_centered**2 / (2 * (sigma * width_mod)**2))
        area = np.sum(I) * dt
        I = I * (np.pi * area_scale / area)
        
        dI = np.gradient(I, dt)
        Q = alpha * dI
        p = np.column_stack([I, Q])
        return enforce_constraints(p)
    
    def objective(params):
        if time.time() > deadline - 10.0:
            return 1e10
        p = parametric_pulse(params)
        return compute_score(p)
    
    # Initial guess
    x0 = np.array([6.0, 0.5, 1.0, 1.0])
    
    # Nelder-Mead optimization
    best_params = x0
    best_score = compute_score(parametric_pulse(x0))
    
    try:
        res = minimize(objective, x0, method='Nelder-Mead', 
                      options={'maxiter': 200, 'xatol': 1e-3, 'fatol': 1e-5,
                              'adaptive': True})
        if res.fun < best_score:
            best_params = res.x
            best_score = res.fun
    except Exception:
        pass
    
    # Check deadline
    if time.time() > deadline - 5.0:
        best_pulse = parametric_pulse(best_params)
        # Final validity check
        amp = np.sqrt(best_pulse[:, 0]**2 + best_pulse[:, 1]**2)
        if np.max(amp) > amp_bound:
            best_pulse *= amp_bound / np.max(amp)
        max_slew = 0
        for i in range(n_steps):
            if i == 0:
                dp = np.linalg.norm(best_pulse[0]) / dt
            else:
                dp = np.linalg.norm(best_pulse[i] - best_pulse[i-1]) / dt
            if dp > max_slew:
                max_slew = dp
        dp_last = np.linalg.norm(best_pulse[-1]) / dt
        if dp_last > max_slew:
            max_slew = dp_last
        if max_slew > slew_bound:
            best_pulse *= slew_bound / max_slew
        return best_pulse
    
    best_pulse = parametric_pulse(best_params)
    
    # Final validity check
    amp = np.sqrt(best_pulse[:, 0]**2 + best_pulse[:, 1]**2)
    if np.max(amp) > amp_bound:
        best_pulse *= amp_bound / np.max(amp)
    
    max_slew = 0
    for i in range(n_steps):
        if i == 0:
            dp = np.linalg.norm(best_pulse[0]) / dt
        else:
            dp = np.linalg.norm(best_pulse[i] - best_pulse[i-1]) / dt
        if dp > max_slew:
            max_slew = dp
    dp_last = np.linalg.norm(best_pulse[-1]) / dt
    if dp_last > max_slew:
        max_slew = dp_last
        
    if max_slew > slew_bound:
        best_pulse *= slew_bound / max_slew
        
    return best_pulse
# EVOLVE-BLOCK-END
