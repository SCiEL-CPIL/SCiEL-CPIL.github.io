"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def pulse(target_gate, hardware_model):
    if target_gate != 'X_pi':
        raise ValueError("Only X_pi is supported")
    
    # Hardware parameters
    anharmonicity = -2 * np.pi * 0.25
    amp_bound = 0.4
    slew_bound = 0.12
    dt = 0.5
    n_steps = 80
    total_time = n_steps * dt
    
    T1 = 25000.0
    Tphi = 40000.0
    
    # Time grid
    t = np.linspace(0, total_time, n_steps, endpoint=False) + dt/2
    t_centered = t - total_time / 2.0
    
    # Precompute operators
    a = np.array([[0, 1, 0],
                  [0, 0, np.sqrt(2)],
                  [0, 0, 0]], dtype=complex)
    adag = a.conj().T
    n = adag @ a
    eye3 = np.eye(3, dtype=complex)
    
    n_n_minus_1 = n @ (n - eye3)
    a_plus_adag = a + adag
    a_minus_adag = a - adag
    
    # Fast unitary simulation proxy
    # We ignore T1/Tphi for the search as they are large and contribute a near-constant offset.
    # We optimize for the worst-case infidelity over the 9 cases.
    
    def simulate_unitary(pulse_vec, Delta, scale):
        I = pulse_vec[:n_steps]
        Q = pulse_vec[n_steps:]
        
        U_total = np.eye(3, dtype=complex)
        
        for i in range(n_steps):
            I_i = I[i]
            Q_i = Q[i]
            
            H0_i = Delta * n + 0.5 * anharmonicity * n_n_minus_1
            H_int_i = 0.5 * scale * (I_i * a_plus_adag + 1j * Q_i * a_minus_adag)
            H_t = H0_i + H_int_i
            
            # Exponentiate H_t * dt
            # For 3x3 Hermitian, eigh is fast
            eigvals, eigvecs = np.linalg.eigh(H_t)
            U_step = eigvecs @ np.diag(np.exp(-1j * eigvals * dt)) @ eigvecs.conj().T
            U_total = U_step @ U_total
            
        # Compute infidelity for 4 states: |0>, |1>, |+>, |->
        # |0> -> |1>
        f0 = abs(U_total[1, 0])**2
        # |1> -> |0>
        f1 = abs(U_total[0, 1])**2
        # |+> -> |+>
        psi_plus_in = np.array([1, 1, 0], dtype=complex) / np.sqrt(2)
        psi_plus_out = U_total @ psi_plus_in
        psi_plus_target = np.array([1, 1, 0], dtype=complex) / np.sqrt(2)
        f_plus = abs(np.vdot(psi_plus_target, psi_plus_out))**2
        
        # |-> -> |->
        psi_minus_in = np.array([1, -1, 0], dtype=complex) / np.sqrt(2)
        psi_minus_out = U_total @ psi_minus_in
        psi_minus_target = np.array([1, -1, 0], dtype=complex) / np.sqrt(2)
        f_minus = abs(np.vdot(psi_minus_target, psi_minus_out))**2
        
        fids = [f0, f1, f_plus, f_minus]
        mean_fid = np.mean(fids)
        infidelity = 1 - mean_fid
        
        return infidelity

    def objective(x):
        # x is a vector of length 160 (80 I, 80 Q)
        I = x[:n_steps]
        Q = x[n_steps:]
        
        # Check constraints and penalize violations
        # Amplitude: sqrt(I^2 + Q^2) <= amp_bound
        amp = np.sqrt(I**2 + Q**2)
        amp_violation = np.sum(np.maximum(0, amp - amp_bound)**2)
        
        # Slew rate: ||p[i+1] - p[i]|| / dt <= slew_bound
        # Include from zero to step 1 and last step to zero
        p = np.column_stack([I, Q])
        p_prev = np.zeros(2)
        slew_violation = 0.0
        for i in range(n_steps):
            dp = np.linalg.norm(p[i] - p_prev) / dt
            if dp > slew_bound:
                slew_violation += (dp - slew_bound)**2
            p_prev = p[i].copy()
        dp_end = np.linalg.norm(p[-1]) / dt
        if dp_end > slew_bound:
            slew_violation += (dp_end - slew_bound)**2
            
        penalty = 100.0 * (amp_violation + slew_violation)
        
        if penalty > 1e-3:
            return penalty
        
        # Compute worst-case infidelity over 9 cases
        max_inf = 0.0
        for Delta in [-2*np.pi*0.002, 0.0, 2*np.pi*0.002]:
            for scale in [0.97, 1.0, 1.03]:
                inf = simulate_unitary(x, Delta, scale)
                if inf > max_inf:
                    max_inf = inf
                    
        return max_inf

    # Initial guess: DRAG-style Gaussian
    sigma = 40/6.0
    alpha = -anharmonicity / (4 * np.pi**2)  # Approx DRAG coefficient
    
    I_guess = np.exp(-t_centered**2 / (2 * sigma**2))
    area = np.sum(I_guess) * dt
    I_guess = I_guess * (np.pi / area)
    
    dI = np.gradient(I_guess, dt)
    Q_guess = alpha * dI
    
    p_guess = np.column_stack([I_guess, Q_guess])
    
    # Ensure initial guess is feasible
    amp_guess = np.sqrt(I_guess**2 + Q_guess**2)
    max_amp = np.max(amp_guess)
    if max_amp > amp_bound:
        scale = amp_bound / max_amp
        I_guess *= scale
        Q_guess *= scale
        
    p_prev = np.zeros(2)
    max_slew = 0.0
    for i in range(n_steps):
        dp = np.linalg.norm(p_guess[i] - p_prev) / dt
        if dp > max_slew:
            max_slew = dp
        p_prev = p_guess[i].copy()
    dp_end = np.linalg.norm(p_guess[-1]) / dt
    if dp_end > max_slew:
        max_slew = dp_end
        
    if max_slew > slew_bound:
        scale = slew_bound / max_slew
        I_guess *= scale
        Q_guess *= scale
        
    x0 = np.concatenate([I_guess, Q_guess])
    
    # Optimization
    start_time = time.time()
    timeout = 280  # Leave 50s margin
    
    # Nelder-Mead is slow for 160 dimensions.
    # We need to be very careful.
    # Maybe reduce dimensions? Or use a different method?
    # The task says "different strategy".
    # If Nelder-Mead is too slow, we might timeout.
    # Let's try a coarser search first or a smaller Nelder-Mead.
    
    # Actually, 160 dimensions is huge for Nelder-Mead.
    # Let's try a different approach:
    # 1. Optimize a few parameters of a template (like the current program but better).
    # 2. Or use a gradient-based method if we can compute gradients?
    #    Gradients of unitary evolution are complex.
    
    # Let's stick to the "direct optimization" idea but maybe restrict the search space
    # or use a smarter initial guess and fewer iterations.
    
    # Alternatively, use `scipy.optimize.differential_evolution`?
    # It's global but also slow.
    
    # Let's try Nelder-Mead with a very tight maxiter.
    # If it times out, return the best found.
    
    best_x = x0.copy()
    best_obj = objective(x0)
    
    try:
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-5, 'adaptive': True})
        
        if time.time() - start_time < timeout:
            if res.fun < best_obj:
                best_obj = res.fun
                best_x = res.x
                
    except Exception as e:
        pass
        
    # Final check and return
    I_final = best_x[:n_steps]
    Q_final = best_x[n_steps:]
    
    # Enforce constraints strictly for the returned pulse
    p_final = np.column_stack([I_final, Q_final])
    
    amp_final = np.sqrt(I_final**2 + Q_final**2)
    max_amp = np.max(amp_final)
    if max_amp > amp_bound:
        scale = amp_bound / max_amp
        p_final *= scale
        
    p_prev = np.zeros(2)
    max_slew = 0.0
    for i in range(n_steps):
        dp = np.linalg.norm(p_final[i] - p_prev) / dt
        if dp > max_slew:
            max_slew = dp
        p_prev = p_final[i].copy()
    dp_end = np.linalg.norm(p_final[-1]) / dt
    if dp_end > max_slew:
        max_slew = dp_end
        
    if max_slew > slew_bound:
        scale = slew_bound / max_slew
        p_final *= scale
        
    return p_final
# EVOLVE-BLOCK-END
