import numpy as np
from scipy.special import roots_genlaguerre
from scipy.optimize import minimize_scalar
import time

def construct():
    start_time = time.time()
    deadline = start_time + 550.0  # Safety margin

    def get_roots(alpha):
        # Get roots of generalized Laguerre polynomial L_12^(alpha)
        # n=12, alpha is the parameter
        x, w = roots_genlaguerre(12, alpha)
        return x

    def condition_number_objective(alpha):
        if alpha < -0.5: # Roots must be positive, alpha > -1 for standard definition, but let's keep it safe
            return 1e10
        try:
            roots = get_roots(alpha)
            # Ensure roots are positive and finite
            if not np.all(np.isfinite(roots)) or not np.all(roots > 0):
                return 1e10
            
            # Construct Vandermonde matrix
            # V[i, j] = roots[i]^j
            # Using powers 0 to 11
            powers = np.arange(12)
            V = roots[:, None] ** powers[None, :]
            
            # Compute condition number
            # Using SVD for stability
            cond = np.linalg.cond(V)
            return cond
        except Exception:
            return 1e10

    # 1. Grid search for alpha
    # The roots spread out as alpha increases. 
    # Typically, alpha around 1-5 might be optimal for balancing spread and decay.
    alpha_grid = np.arange(0, 10, 0.1)
    best_alpha = 0.0
    best_cond = condition_number_objective(0.0)
    
    for a in alpha_grid:
        if time.time() > deadline - 60:
            break
        cond = condition_number_objective(a)
        if cond < best_cond:
            best_cond = cond
            best_alpha = a

    # 2. Refine with scalar minimization
    # Use the best grid point as the center for a local search
    bounds = (max(0, best_alpha - 1.0), best_alpha + 1.0)
    
    try:
        res = minimize_scalar(condition_number_objective, bounds=bounds, method='bounded', 
                              options={'xatol': 1e-6})
        if res.success and res.fun < best_cond:
            best_alpha = res.x
            best_cond = res.fun
    except Exception:
        pass

    # 3. Final check and return
    # Ensure we are within time
    if time.time() > deadline - 10:
        roots = get_roots(best_alpha)
        return [float(x) for x in roots]

    roots = get_roots(best_alpha)
    # Sort roots to be consistent
    roots = np.sort(roots)
    
    return [float(x) for x in roots]
