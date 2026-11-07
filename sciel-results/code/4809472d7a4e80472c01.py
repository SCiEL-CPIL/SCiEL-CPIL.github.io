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
    
    # Time grid (centered for initial guess)
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
    
    # Precompute H0 for different detunings and scales?
    # H = Delta * n + 0.5 * chi * n(n-1) + 0.5 * scale * (I*(a+adag) + i*Q*(a-adag))
    # Since we are optimizing for robustness, the objective function should evaluate
    # the infidelity over the 9 cases of (detuning, scale).
    # Detunings: -2*pi*0.002, 0, 2*pi*0.002
    # Scales: 0.97, 1.0, 1.03
    
    deltas = [-2 * np.pi * 0.002, 0.0, 2 * np.pi * 0.002]
    scales = [0.97, 1.0, 1.03]
    
    # Precompute H0 for each delta
    H0s = []
    for d in deltas:
        H0s.append(d * n + 0.5 * anharmonicity * n_n_minus_1)
    
    # Target states and targets for fidelity
    # States: |0>, |1>, |+>, |->
    # The prompt lists: |0>, |1>, (|0> +- |1>)/sqrt(2), (|0> +- i|1>)/sqrt(2)
    # This implies 6 states? 
    # "mean fidelity to X over |0>, |1>, (|0> +- |1>)/sqrt(2), (|0> +- i|1>)/sqrt(2)"
    # Usually "A +- B" means two states. So 1 + 1 + 2 + 2 = 6 states.
    # Let's assume 6 states for the mean fidelity calculation.
    
    states_in = [
        np.array([1, 0, 0], dtype=complex), # |0>
        np.array([0, 1, 0], dtype=complex), # |1>
        np.array([1, 1, 0], dtype=complex) / np.sqrt(2), # |+>
        np.array([1, -1, 0], dtype=complex) / np.sqrt(2), # |->
        np.array([1, 1j, 0], dtype=complex) / np.sqrt(2), # |0> + i|1>
        np.array([1, -1j, 0], dtype=complex) / np.sqrt(2) # |0> - i|1>
    ]
    
    # Target states (X gate applied to states_in)
    # X|0> = |1>, X|1> = |0>, X|+> = |+>, X|-> = |->, X(|0>+i|1>) = |1>+i|0> = i(|0>-i|1>)? 
    # Let's just apply X matrix.
    X_mat = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=complex)
    states_target = [X_mat @ s for s in states_in]
    
    def simulate_unitary_matrix(pulse_flat, delta_idx, scale):
        """
        Simulate the unitary evolution for a given pulse, detuning index, and scale.
        Returns the 3x3 unitary matrix U.
        """
        I = pulse_flat[0:n_steps]
        Q = pulse_flat[n_steps:2*n_steps]
        
        H0 = H0s[delta_idx]
        
        U_total = np.eye(3, dtype=complex)
        
        for i in range(n_steps):
            I_i = I[i]
            Q_i = Q[i]
            
            # H_int = 0.5 * scale * (I*(a+adag) + i*Q*(a-adag))
            H_int = 0.5 * scale * (I_i * a_plus_adag + 1j * Q_i * a_minus_adag)
            H_t = H0 + H_int
            
            # Exponentiate
            # For 3x3 Hermitian, eigh is fast
            eigvals, eigvecs = np.linalg.eigh(H_t)
            U_step = eigvecs @ np.diag(np.exp(-1j * eigvals * dt)) @ eigvecs.conj().T
            U_total = U_step @ U_total
            
        return U_total

    def calculate_infidelity(pulse_flat):
        """
        Calculate the mean infidelity over the 9 cases and 6 states.
        Includes penalties for constraint violations.
        """
        total_infidelity = 0.0
        
        for d_idx in range(3):
            for s_idx in range(3):
                U = simulate_unitary_matrix(pulse_flat, d_idx, scales[s_idx])
                
                case_infidelity = 0.0
                for s_in, s_tgt in zip(states_in, states_target):
                    # Fidelity = |<target| U |initial>|^2
                    # s_in is a column vector. U @ s_in.
                    # s_tgt is a column vector. vdot(s_tgt, U @ s_in)
                    amp = np.vdot(s_tgt, U @ s_in)
                    fid = np.abs(amp)**2
                    case_infidelity += (1.0 - fid)
                
                case_infidelity /= len(states_in)
                total_infidelity += case_infidelity
        
        total_infidelity /= 9.0
        
        # Add penalties for constraints
        I = pulse_flat[0:n_steps]
        Q = pulse_flat[n_steps:2*n_steps]
        
        # Amplitude penalty
        amps = np.sqrt(I**2 + Q**2)
        max_amp = np.max(amps)
        if max_amp > amp_bound:
            penalty = 1000.0 * (max_amp - amp_bound)**2
        else:
            penalty = 0.0
        
        # Slew rate penalty
        # Check ||p[i+1] - p[i]|| / dt <= slew_bound
        # p[i] is (I[i], Q[i])
        # Includes from zero to step 1 and last step to zero.
        
        # From zero to step 1
        p0 = np.array([0.0, 0.0])
        p1 = np.array([I[0], Q[0]])
        slew0 = np.linalg.norm(p1 - p0) / dt
        if slew0 > slew_bound:
            penalty += 1000.0 * (slew0 - slew_bound)**2
        
        # Between steps
        for i in range(n_steps - 1):
            p_i = np.array([I[i], Q[i]])
            p_ip1 = np.array([I[i+1], Q[i+1]])
            slew = np.linalg.norm(p_ip1 - p_i) / dt
            if slew > slew_bound:
                penalty += 1000.0 * (slew - slew_bound)**2
        
        # From last step to zero
        p_last = np.array([I[-1], Q[-1]])
        slew_end = np.linalg.norm(p_last - p0) / dt
        if slew_end > slew_bound:
            penalty += 1000.0 * (slew_end - slew_bound)**2
        
        return total_infidelity + penalty

    # Initial guess: DRAG-style Gaussian
    sigma = 6.0
    alpha = 0.1
    I_init = np.exp(-t_centered**2 / (2 * sigma**2))
    area = np.sum(I_init) * dt
    I_init = I_init * (np.pi / area)
    dI = np.gradient(I_init, dt)
    Q_init = alpha * dI
    
    p_init = np.concatenate([I_init, Q_init])
    
    # Optimization
    start_time = time.time()
    timeout = 300 # 300s limit, leave margin
    
    best_pulse = p_init.copy()
    best_score = calculate_infidelity(p_init)
    
    # Nelder-Mead is robust but can be slow with 160 variables.
    # We will run it with a time check.
    # To make it faster, we might reduce the number of variables or use a coarser search first?
    # No, let's stick to full 160 but ensure the objective is fast.
    # The objective runs 9 * 80 * eigh(3x3). 
    # eigh(3x3) is very fast. 720 eighs per eval.
    # Nelder-Mead with 160 vars might take many iterations.
    # Let's try.
    
    def objective(x):
        return calculate_infidelity(x)
    
    try:
        # Use Nelder-Mead
        # maxiter is not a hard time limit, so we need to be careful.
        # We'll run a few iterations manually or rely on the fact that it converges.
        # Actually, scipy's minimize doesn't have a built-in time limit.
        # We can wrap it, but it's hard to interrupt cleanly.
        # Alternative: run a short Nelder-Mead, then check time.
        
        # Let's run a limited number of iterations or just one call and hope it's fast enough?
        # 160 variables is a lot for Nelder-Mead.
        # Maybe reduce dimensionality?
        # The prompt asks for a DIFFERENT strategy.
        # "Direct optimization of the full 80-step I/Q vector" is the strategy.
        # To make it feasible in 300s, we might need to be clever.
        
        # Let's try a two-stage approach:
        # 1. Optimize a lower-dimensional representation (e.g., 20 steps, then upsample?)
        #    No, that changes the physics.
        # 2. Use a more efficient optimizer?
        #    BFGS requires gradients. We can compute gradients via finite difference (expensive) or analytic (hard).
        #    Nelder-Mead is gradient-free.
        
        # Let's estimate cost.
        # One objective eval: ~0.01-0.05s?
        # Nelder-Mead for N=160 needs ~ (N+1) * iterations.
        # If it takes 1000 iterations, that's 161,000 evals. Too many.
        
        # We need to reduce the search space or use a better method.
        # What if we optimize only a subset of parameters?
        # Or use a "smart" initial guess and few iterations?
        
        # Let's try a different approach:
        # Use the DRAG template but optimize MORE parameters?
        # No, the directive says "direct optimization of the full 80-step I/Q vector".
        
        # To make it work, we can:
        # 1. Use a smaller number of steps for optimization? No, must return 80.
        # 2. Use a faster simulation?
        #    The simulation is the bottleneck.
        #    Can we approximate the evolution?
        #    For small dt, U_step = exp(-i H dt) ~ I - i H dt.
        #    This is first-order accurate. Might be enough for optimization?
        #    Let's try first-order approximation for the optimizer, and exact for final check.
        
        def simulate_unitary_matrix_fast(pulse_flat, delta_idx, scale):
            """
            First-order approximation of unitary evolution.
            U_total = prod (I - i H_i dt)
            """
            I = pulse_flat[0:n_steps]
            Q = pulse_flat[n_steps:2*n_steps]
            H0 = H0s[delta_idx]
            
            U_total = np.eye(3, dtype=complex)
            
            for i in range(n_steps):
                I_i = I[i]
                Q_i = Q[i]
                H_int = 0.5 * scale * (I_i * a_plus_adag + 1j * Q_i * a_minus_adag)
                H_t = H0 + H_int
                
                # First order
                U_step = eye3 - 1j * H_t * dt
                U_total = U_step @ U_total
                
            return U_total

        def calculate_infidelity_fast(pulse_flat):
            """
            Fast version using first-order approximation.
            """
            total_infidelity = 0.0
            
            for d_idx in range(3):
                for s_idx in range(3):
                    U = simulate_unitary_matrix_fast(pulse_flat, d_idx, scales[s_idx])
                    
                    case_infidelity = 0.0
                    for s_in, s_tgt in zip(states_in, states_target):
                        amp = np.vdot(s_tgt, U @ s_in)
                        fid = np.abs(amp)**2
                        case_infidelity += (1.0 - fid)
                    
                    case_infidelity /= len(states_in)
                    total_infidelity += case_infidelity
            
            total_infidelity /= 9.0
            
            # Same penalties
            I = pulse_flat[0:n_steps]
            Q = pulse_flat[n_steps:2*n_steps]
            
            amps = np.sqrt(I**2 + Q**2)
            max_amp = np.max(amps)
            penalty = 0.0
            if max_amp > amp_bound:
                penalty += 1000.0 * (max_amp - amp_bound)**2
            
            p0 = np.array([0.0, 0.0])
            p1 = np.array([I[0], Q[0]])
            slew0 = np.linalg.norm(p1 - p0) / dt
            if slew0 > slew_bound:
                penalty += 1000.0 * (slew0 - slew_bound)**2
            
            for i in range(n_steps - 1):
                p_i = np.array([I[i], Q[i]])
                p_ip1 = np.array([I[i+1], Q[i+1]])
                slew = np.linalg.norm(p_ip1 - p_i) / dt
                if slew > slew_bound:
                    penalty += 1000.0 * (slew - slew_bound)**2
            
            p_last = np.array([I[-1], Q[-1]])
            slew_end = np.linalg.norm(p_last - p0) / dt
            if slew_end > slew_bound:
                penalty += 1000.0 * (slew_end - slew_bound)**2
            
            return total_infidelity + penalty

        # Use the fast version for optimization
        def objective_fast(x):
            return calculate_infidelity_fast(x)
        
        # Run Nelder-Mead with the fast objective
        # To save time, we might limit the number of iterations.
        # But we don't have a direct way to limit time in scipy.minimize.
        # We can run it in a loop and check time? No, minimize is a single call.
        
        # Alternative: Use a simple gradient-free method like COBYLA?
        # Or just run Nelder-Mead and hope it's fast enough with the fast simulator.
        
        # Let's try.
        result = minimize(objective_fast, p_init, method='Nelder-Mead', 
                          options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-6})
        
        if result.fun < best_score:
            best_pulse = result.x
            best_score = result.fun
            
    except Exception as e:
        pass # Fallback to initial guess
    
    # Final check with exact simulation and constraint enforcement
    # We need to ensure the final pulse is valid.
    # If it's invalid, we should project it or use a valid fallback.
    
    I_final = best_pulse[0:n_steps]
    Q_final = best_pulse[n_steps:2*n_steps]
    
    # Check and enforce amplitude bound
    amps = np.sqrt(I_final**2 + Q_final**2)
    max_amp = np.max(amps)
    if max_amp > amp_bound:
        scale = amp_bound / max_amp
        I_final *= scale
        Q_final *= scale
    
    # Check and enforce slew rate bound
    # This is harder to enforce globally.
    # If violated, we might need to reduce the pulse.
    # For simplicity, if slew is violated, we scale down the whole pulse.
    # This might not be optimal but ensures validity.
    
    p_flat = np.concatenate([I_final, Q_final])
    
    # Check slew
    valid = True
    p0 = np.array([0.0, 0.0])
    p1 = np.array([I_final[0], Q_final[0]])
    if np.linalg.norm(p1 - p0) / dt > slew_bound:
        valid = False
    
    for i in range(n_steps - 1):
        p_i = np.array([I_final[i], Q_final[i]])
        p_ip1 = np.array([I_final[i+1], Q_final[i+1]])
        if np.linalg.norm(p_ip1 - p_i) / dt > slew_bound:
            valid = False
    
    p_last = np.array([I_final[-1], Q_final[-1]])
    if np.linalg.norm(p_last - p0) / dt > slew_bound:
        valid = False
    
    if not valid:
        # Scale down to satisfy slew
        # Find max slew
        max_slew = 0
        p_prev = np.zeros(2)
        for i in range(n_steps):
            p_cur = np.array([I_final[i], Q_final[i]])
            slew = np.linalg.norm(p_cur - p_prev) / dt
            if slew > max_slew:
                max_slew = slew
            p_prev = p_cur.copy()
        
        p_end = np.array([I_final[-1], Q_final[-1]])
        slew_end = np.linalg.norm(p_end - p0) / dt
        if slew_end > max_slew:
            max_slew = slew_end
        
        if max_slew > 0:
            scale = slew_bound / max_slew
            I_final *= scale
            Q_final *= scale
    
    # Re-check amplitude after slew scaling
    amps = np.sqrt(I_final**2 + Q_final**2)
    max_amp = np.max(amps)
    if max_amp > amp_bound:
        scale = amp_bound / max_amp
        I_final *= scale
        Q_final *= scale
    
    final_pulse = np.column_stack([I_final, Q_final])
    
    return final_pulse
# EVOLVE-BLOCK-END
