import numpy as np
import time
import random

def construct():
    N = 24
    # 1. Setup evaluation
    # Use a moderate grid for SA speed, refine at the end
    M_sa = 2048 # Power of 2 for FFT efficiency if we used it, but we use matmul
    theta_sa = np.linspace(0, 2 * np.pi, M_sa, endpoint=False)
    z_sa = np.exp(1j * theta_sa)
    V_sa = np.vander(z_sa, N=N, increasing=True)
    
    M_final = 20000
    theta_final = np.linspace(0, 2 * np.pi, M_final, endpoint=False)
    z_final = np.exp(1j * theta_final)
    V_final = np.vander(z_final, N=N, increasing=True)

    def fast_eval(c):
        """Approximate sup norm using coarse grid."""
        p_vals = V_sa @ c
        return np.max(np.abs(p_vals))

    def slow_eval(c):
        """High precision sup norm with refinement."""
        p_vals = V_final @ c
        mags = np.abs(p_vals)
        idx_max = np.argmax(mags)
        max_val = mags[idx_max]
        
        # Refine top peaks
        # Find local maxima indices
        # Simple approach: check neighbors of the global max
        # Since N is small, peaks are isolated.
        h = 2 * np.pi / M_final
        
        # We'll refine the global max and maybe the 2nd highest if distinct
        # For SA, we mostly use fast_eval. This is for final candidates.
        
        def refine_peak(theta_center):
            # Golden section or ternary search?
            # Let's use scipy's bounded minimization for -|P(z)|
            from scipy.optimize import minimize_scalar
            
            def obj(t):
                z_loc = np.exp(1j * (theta_center + t))
                # Compute P(z_loc)
                # Using Horner's or direct sum
                # Direct sum is fine for N=24
                z_powers = np.ones(N, dtype=complex)
                for k in range(1, N):
                    z_powers[k] = z_powers[k-1] * z_loc
                val = np.dot(c, z_powers)
                return -np.abs(val)
            
            # Bounds [-h, h]
            # Handle wrap around
            # If theta_center is near 0 or 2pi, we might cross.
            # But h is small (2pi/20000 ~ 3e-4).
            # If idx is 0, range is [-h, h] -> [2pi-h, h]. 
            # minimize_scalar bounded handles linear domain.
            # We just need to ensure the function is continuous in the domain.
            # P(e^{i(t)}) is periodic.
            # So minimizing in [-h, h] around 0 is equivalent to [2pi-h, h] in the circle.
            # The function value is the same.
            res = minimize_scalar(obj, bounds=(-h, h), method='bounded', options={'xatol': 1e-10})
            if res.success:
                return -res.fun
            return max_val # fallback
        
        refined_max = refine_peak(theta_final[idx_max])
        
        # Check if there's another significant peak
        # Sort indices by magnitude, pick top 3 distinct
        # For N=24, usually 1 or 2 main peaks.
        # To be safe and fast, just refine the top 1 and 2 if they are far apart.
        # Actually, let's just refine the top 1. The global max is what matters.
        # If the 2nd highest is very close, it might be the true max.
        # Let's refine top 2 distinct peaks.
        
        # Get top 2 indices
        # np.argsort is expensive. 
        # Just check the max and the max of the rest?
        # Or just refine the max. If the second max is > 0.99 * max, refine it too.
        
        mags_sorted_idx = np.argsort(mags)[::-1]
        refined_vals = [refined_max]
        for idx in mags_sorted_idx[1:3]:
            # Check if this peak is in a different lobe
            # Distance in theta
            dist = np.abs(theta_final[idx] - theta_final[idx_max])
            # Circular distance
            dist = min(dist, 2*np.pi - dist)
            if dist > 0.1: # Arbitrary threshold, peaks are usually well separated
                 refined_vals.append(refine_peak(theta_final[idx]))
        
        return max(refined_vals)

    # 2. Simulated Annealing
    start_time = time.time()
    max_time = 115.0
    
    # Initial solution: Random
    c = np.random.choice([-1, 1], N).astype(float)
    best_c = c.copy()
    best_score = slow_eval(best_c) # Use slow eval for initial baseline? 
    # Actually, fast_eval is better for initialization speed, but we need comparable scores.
    # Let's use fast_eval for the SA loop, and slow_eval for the final best.
    # But we need to track the "best" based on fast_eval, then polish.
    
    current_score = fast_eval(c)
    best_fast_score = current_score
    best_c = c.copy()
    
    T = 1.0 # Initial temperature
    T_min = 1e-4
    cooling = 0.999 # Slow cooling for many iterations
    
    # Estimate number of iterations
    # Each iter: 1 fast_eval (matmul 2048x24) ~ very fast.
    # We can do ~100,000 iterations in 120s?
    # Matmul 2048x24 is ~50k FLOPs. Python overhead is higher.
    # Let's target 50,000 iterations.
    
    while time.time() - start_time < max_time:
        # Propose move: flip one coefficient
        k = random.randint(0, N-1)
        old_val = c[k]
        c[k] = -old_val
        
        new_score = fast_eval(c)
        delta = new_score - current_score
        
        if delta < 0 or random.random() < np.exp(-delta / T):
            current_score = new_score
            if new_score < best_fast_score:
                best_fast_score = new_score
                best_c = c.copy()
        else:
            c[k] = old_val # Revert
            
        T *= cooling
        if T < T_min:
            T = T_min
            
    # 3. Local Search Polish on the best found
    # Use slow_eval for the final polish to ensure accuracy
    current_c = best_c.copy()
    current_slow_score = slow_eval(current_c)
    
    improved = True
    while improved and (time.time() - start_time) < max_time:
        improved = False
        for k in range(N):
            old_val = current_c[k]
            current_c[k] = -old_val
            new_slow_score = slow_eval(current_c)
            
            if new_slow_score < current_slow_score - 1e-12:
                current_slow_score = new_slow_score
                improved = True
            else:
                current_c[k] = old_val

    # Return the best coefficients
    return [int(x) for x in current_c]
