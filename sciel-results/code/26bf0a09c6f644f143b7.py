"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

def find_root_system(F, x0):
    """
    Find a root x in R^2 such that ||F(x)|| <= 1e-8.
    
    Strategy:
    1. Use differential_evolution for global robustness (derivative-free).
    2. Use Nelder-Mead for local polishing (derivative-free).
    3. Strict call counting and deadline management.
    """
    x0 = np.array(x0, dtype=float)
    
    # Deadline guard: 120s limit, keep 10s margin for safety/overhead
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Call counter to track usage
    call_count = 0
    max_calls_allowed = 450  # Stay well below 500 to avoid failure penalty
    
    def objective(x):
        nonlocal call_count
        if time.time() > deadline:
            raise TimeoutError("Deadline exceeded")
        if call_count >= max_calls_allowed:
            raise RuntimeError("Call limit exceeded")
        
        try:
            fx = F(x)
            call_count += 1
            fx = np.asarray(fx, dtype=float)
            if not np.all(np.isfinite(fx)):
                return 1e100
            return np.sum(fx**2)
        except Exception:
            call_count += 1
            return 1e100

    best_x = x0.copy()
    best_val = np.inf

    def update_best(x, val):
        nonlocal best_x, best_val
        if val < best_val:
            best_val = val
            best_x = x.copy()

    try:
        # Phase 1: Global Search with Differential Evolution
        # Bounds: We don't know the scale, so use a wide box around x0.
        # If x0 is 0, use [-10, 10]. If x0 is large, scale accordingly.
        scale = max(1.0, np.max(np.abs(x0)) * 10)
        bounds = [(x0[0] - scale, x0[0] + scale), (x0[1] - scale, x0[1] + scale)]
        
        # DE parameters:
        # maxiter: limit iterations to control calls. 
        # Each iter uses popsize * 2 calls roughly. 
        # popsize=10, maxiter=20 -> ~400 calls. 
        # Let's be conservative: popsize=5, maxiter=10 -> ~100 calls.
        # We want to leave room for Nelder-Mead.
        
        res_de = differential_evolution(
            objective, 
            bounds, 
            maxiter=20, 
            popsize=5, 
            tol=1e-6, 
            polish=False, 
            init='sobol'
        )
        
        if res_de.success or res_de.fun < best_val:
            update_best(res_de.x, res_de.fun)
            
    except Exception:
        pass

    # Phase 2: Local Polish with Nelder-Mead
    # Nelder-Mead is robust and requires no derivatives.
    if time.time() < deadline and call_count < max_calls_allowed:
        try:
            # Initialize simplex around best_x
            # If best_val is inf, start from x0
            start_x = best_x if np.isfinite(best_val) else x0
            
            # Small initial step for NM
            step = 1e-2
            x0_nm = start_x
            # Nelder-Mead options:
            # maxiter: limit function calls. 
            # maxfev: explicit limit on function evaluations.
            # We have max_calls_allowed - call_count remaining.
            remaining_calls = max_calls_allowed - call_count
            if remaining_calls > 10:
                res_nm = minimize(
                    objective, 
                    x0_nm, 
                    method='Nelder-Mead', 
                    options={
                        'xatol': 1e-12, 
                        'fatol': 1e-12, 
                        'maxiter': 100, 
                        'maxfev': remaining_calls
                    }
                )
                if res_nm.fun < best_val:
                    update_best(res_nm.x, res_nm.fun)
                    
        except Exception:
            pass

    # Final check: ensure we return a valid array
    if not np.all(np.isfinite(best_x)):
        return x0
    return best_x

# EVOLVE-BLOCK-END
