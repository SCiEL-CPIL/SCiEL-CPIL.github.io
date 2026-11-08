"""Rewritten program for robust control pulses for a three-level transmon."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.linalg import expm
from scipy.optimize import minimize

def pulse(target_gate, hardware_model):
    """
    Generate a robust pulse for X_pi gate using fast model-based optimization.
    
    Strategy:
    1. Pre-compute superoperator components to speed up simulation.
    2. Parametric DRAG-style pulse optimization.
    3. Deadline guard to ensure completion within 300s.
    """
    if target_gate != 'X_pi':
        raise ValueError("Only X_pi is supported")
    
    start_time = time.time()
    deadline = 280.0  # seconds, safety margin
    
    # Hardware parameters
    anharmonicity = -2 * np.pi * 0.25
    amp_bound = 0.4
    slew_bound = 0.12
    dt = 0.5
    n_steps = 80
    T1 = 25000.0
    Tphi = 40000.0
    
    # Time points
    t = np.linspace(0, n_steps * dt, n_steps, endpoint=False) + dt/2
    t_centered = t - n_steps * dt / 2.0
    
    # Pre-compute matrices
    a = np.array([[0, 1, 0],
                  [0, 0, np.sqrt(2)],
                  [0, 0, 0]], dtype=complex)
    adag = a.conj().T
    n_op = adag @ a
    eye3 = np.eye(3, dtype=complex)
    
    # Lindblad operators
    L1 = a / np.sqrt(T1)
    L2 = n_op * np.sqrt(2.0 / Tphi)
    
    def lindblad_superop(L):
        LdagL = L.conj().T @ L
        return np.kron(L.conj().T.T, L) - 0.5 * (np.kron(eye3, LdagL) - np.kron(LdagL.T, eye3))
    
    S_diss = lindblad_superop(L1) + lindblad_superop(L2)
    
    def commutator_superop(H):
        return -1j * (np.kron(eye3, H) - np.kron(H.T, eye3))
    
    # Hamiltonian components
    H_anh = 0.5 * anharmonicity * n_op @ (n_op - eye3)
    H_I_term = 0.5 * (a + adag)  # Multiplied by I * scale
    H_Q_term = 0.5j * (a - adag) # Multiplied by Q * scale
    
    # Pre-compute superoperator components
    # S_k = S_diss - i [H_k, .]
    # H_k = delta * n_op + H_anh + scale * (I_k * H_I_term + Q_k * H_Q_term)
    # S_k = S_diss - i [delta*n_op + H_anh, .] - i * scale * (I_k [H_I_term, .] + Q_k [H_Q_term, .])
    
    S_base_I = commutator_superop(H_I_term)
    S_base_Q = commutator_superop(H_Q_term)
    S_n = commutator_superop(n_op)
    S_anh = commutator_superop(H_anh)
    
    # Test states
    psi_states = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0] / np.sqrt(2),
        [1, -1, 0] / np.sqrt(2),
        [1, 1j, 0] / np.sqrt(2),
        [1, -1j, 0] / np.sqrt(2)
    ], dtype=complex)
    
    rho_initial = np.array([np.outer(p, p.conj()) for p in psi_states])
    rho_initial_vec = np.array([rho.flatten() for rho in rho_initial]) # (6, 9)
    
    X = np.array([[0, 1, 0],
                  [1, 0, 0],
                  [0, 0, 0]], dtype=complex)
    
    psi_target = np.array([X @ p for p in psi_states], dtype=complex)
    
    # 9 cases
    deltas = [-2 * np.pi * 0.002, 0, 2 * np.pi * 0.002]
    scales = [0.97, 1.0, 1.03]
    
    # Pre-compute case-specific base superoperators
    # S_k = (S_diss - i[delta*n + H_anh, .]) + scale * (I_k * S_I + Q_k * S_Q)
    # Let M_case = S_diss - i[delta*n + H_anh, .]
    # Then S_k = M_case + scale * (I_k * S_base_I + Q_k * S_base_Q)
    
    M_cases = []
    for d in deltas:
        H_det = d * n_op + H_anh
        M = S_diss + commutator_superop(H_det)
        M_cases.append(M)
    
    def simulate_pulse(I, Q):
        """
        Simulate the pulse and return the score (mean + max infidelity).
        """
        infidelities = []
        
        for idx, M_case in enumerate(M_cases):
            scale = scales[idx % 3]
            
            rho_vecs = rho_initial_vec.copy() # (6, 9)
            
            for k in range(n_steps):
                # Construct S_k
                # S_k = M_case + scale * (I_k * S_base_I + Q_k * S_base_Q)
                S_k = M_case + scale * (I[k] * S_base_I + Q[k] * S_base_Q)
                
                # Exponentiate
                U = expm(S_k * dt)
                
                # Apply to all states
                # rho_vecs is (6, 9), U is (9, 9)
                # We want rho_new[i] = U @ rho_old[i]
                # rho_vecs.T is (9, 6)
                rho_vecs = (U @ rho_vecs.T).T
            
            # Compute fidelities
            # rho_out[i] is the i-th row of rho_vecs reshaped to (3,3)
            # F_i = <psi_target[i] | rho_out[i] | psi_target[i]>
            # = sum_{mn} psi_t[i,m]^* rho[m,n] psi_t[i,n]
            # Using vectorization: F = psi_t^* @ (rho_vec @ psi_t) ? No.
            # vec(rho) = [r00, r01, r02, r10, ...]
            # rho @ psi = [sum_j r0j p_j, sum_j r1j p_j, ...]
            # This is easier to compute by reshaping.
            
            fids = []
            for i in range(6):
                rho_out = rho_vecs[i].reshape(3, 3)
                # Fidelity
                fid = np.vdot(psi_target[i], rho_out @ psi_target[i]).real
                fids.append(fid)
            
            mean_fid = np.mean(fids)
            infidelities.append(1.0 - mean_fid)
        
        infidelities = np.array(infidelities)
        return np.mean(infidelities) + np.max(infidelities)
    
    def enforce_constraints(p):
        p = p.copy()
        # Amplitude
        amp = np.sqrt(p[:, 0]**2 + p[:, 1]**2)
        max_amp = np.max(amp)
        if max_amp > amp_bound:
            p *= amp_bound / max_amp
        
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
    
    def parametric_pulse(params):
        sigma, area_scale, alpha, t_center = params
        # Gaussian envelope
        t_rel = t - t_center
        I = np.exp(-t_rel**2 / (2 * sigma**2))
        # Normalize area to pi
        area = np.sum(I) * dt
        I = I * (np.pi * area_scale / area)
        
        # DRAG
        dI = np.gradient(I, dt)
        Q = alpha * dI
        
        p = np.column_stack([I, Q])
        return enforce_constraints(p)
    
    def objective(params):
        p = parametric_pulse(params)
        I, Q = p[:, 0], p[:, 1]
        return simulate_pulse(I, Q)
    
    # Initial guess
    x0 = np.array([6.0, 1.0, 0.5, 0.0])
    
    best_params = x0.copy()
    best_score = objective(x0)
    
    # Optimization
    try:
        res = minimize(objective, x0, method='Nelder-Mead', 
                      options={'maxiter': 1000, 'xatol': 1e-4, 'fatol': 1e-6,
                              'adaptive': True})
        if res.fun < best_score:
            best_score = res.fun
            best_params = res.x
    except Exception:
        pass
    
    # Check deadline
    if time.time() - start_time > deadline:
        p = parametric_pulse(best_params)
        return p
    
    # Local search / restarts
    np.random.seed(42)
    for i in range(50):
        if time.time() - start_time > deadline:
            break
        pert = np.random.normal(0, 0.5, size=4)
        pert[0] = max(pert[0], 2.0) # sigma must be positive
        new_params = best_params + pert
        new_score = objective(new_params)
        if new_score < best_score:
            best_score = new_score
            best_params = new_params
    
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
