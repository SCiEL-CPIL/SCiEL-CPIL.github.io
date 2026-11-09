"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    """
    Return 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    Score is the upper bound on C4 (lower is better).
    """
    t0 = time.time()
    DEADLINE = 580  # safety margin before 600s kill
    
    m = 12
    n = 1
    
    # Start with Laguerre roots as initial guess
    try:
        from scipy.special import roots_laguerre
        x_lag, _ = roots_laguerre(m)
        r0 = np.array(x_lag, dtype=np.float64)
    except Exception:
        r0 = np.array([4.0 * (i + 1) for i in range(m)], dtype=np.float64)
    
    # Define the C4 bound function to minimize
    # C4 = max_{x >= 0} |x^n * prod_{i=1}^m (x - r_i)^2 * exp(-x)|
    # For n=1: C4 = max_{x >= 0} x * prod_{i=1}^m (x - r_i)^2 * exp(-x)
    
    def c4_bound(r):
        """Compute the C4 upper bound for given roots r using vectorization."""
        # Create a grid. We use a moderate resolution to balance speed and accuracy.
        # The maximum is likely near the roots or between them.
        x_max = r[-1] * 2.0 + 20.0
        # 2000 points is usually sufficient for this smooth function with local maxes
        x_grid = np.linspace(1e-8, x_max, 2000)
        
        # Vectorized computation of log f(x)
        # log f(x) = log(x) - x + 2 * sum(log|x - r_i|)
        
        # x_grid shape: (N,)
        # r shape: (M,)
        # diff shape: (N, M)
        diff = np.abs(x_grid[:, np.newaxis] - r[np.newaxis, :])
        # Clip to avoid log(0)
        diff = np.maximum(diff, 1e-300)
        
        log_f = np.log(x_grid) - x_grid + 2.0 * np.sum(np.log(diff), axis=1)
        
        # Find max
        max_log_f = np.max(log_f)
        
        # If max log f is very large, exp might overflow, but we just need the relative value
        # for optimization. However, the score is the raw value.
        # If max_log_f > 700, exp overflows. 
        # Given the nature of Laguerre roots, the value shouldn't be astronomically high 
        # unless roots are very close to 0 or each other.
        # To be safe, if it's too high, return a large number.
        if max_log_f > 700:
            return 1e300
        if max_log_f < -700:
            return 0.0
            
        return np.exp(max_log_f)
    
    # Optimize roots to minimize C4
    best_r = r0.copy()
    best_c4 = c4_bound(r0)
    
    # Use Nelder-Mead optimization
    try:
        from scipy.optimize import minimize
        
        # Use log-parameters to ensure positivity
        log_r0 = np.log(r0)
        
        # Count evaluations to enforce a hard stop if time is tight
        eval_count = 0
        
        def objective(log_r):
            nonlocal eval_count
            eval_count += 1
            # Check time every few evaluations
            if eval_count % 10 == 0:
                if time.time() - t0 > DEADLINE:
                    return best_c4
            r = np.exp(log_r)
            # Ensure sorted and positive
            r = np.sort(r)
            if np.any(r <= 0) or r[0] < 1e-10:
                return 1e30
            return c4_bound(r)
        
        # Reduced maxiter to ensure we don't run away
        result = minimize(
            objective,
            log_r0,
            method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
        )
        
        if result.fun < best_c4:
            best_r = np.sort(np.exp(result.x))
            best_c4 = result.fun
        
    except Exception:
        pass
    
    # Check deadline
    if time.time() - t0 > DEADLINE:
        return [float(x) for x in best_r]
    
    # Try a few more local optimizations from different starting points
    try:
        from scipy.optimize import minimize
        
        # Try starting from slightly perturbed Laguerre roots
        for trial in range(3):
            if time.time() - t0 > DEADLINE - 10: # Stop early to allow return
                break
            
            # Perturb the roots
            perturbation = 0.1 * np.random.randn(m)
            r_pert = r0 + perturbation
            r_pert = np.sort(np.maximum(r_pert, 0.1))
            
            log_r_start = np.log(r_pert)
            
            eval_count2 = 0
            def objective2(log_r):
                nonlocal eval_count2
                eval_count2 += 1
                if eval_count2 % 10 == 0:
                    if time.time() - t0 > DEADLINE:
                        return best_c4
                r = np.exp(log_r)
                r = np.sort(r)
                if np.any(r <= 0) or r[0] < 1e-10:
                    return 1e30
                return c4_bound(r)
            
            result2 = minimize(
                objective2,
                log_r_start,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
            )
            
            if result2.fun < best_c4:
                best_r = np.sort(np.exp(result2.x))
                best_c4 = result2.fun
                
    except Exception:
        pass
    
    # Final check
    if time.time() - t0 > DEADLINE:
        return [float(x) for x in best_r]
    
    return [float(x) for x in best_r]
# EVOLVE-BLOCK-END
