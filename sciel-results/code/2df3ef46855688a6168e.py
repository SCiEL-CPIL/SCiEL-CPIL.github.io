import numpy as np
import scipy.optimize as so
import scipy.special as sp
import time

def construct():
    start_time = time.time()
    timeout = 500.0 # Safety margin
    
    # Quadrature nodes and weights
    # Degree of P^2 is 48. Need n such that 2n-1 >= 48 -> n >= 25.
    # Use n=50 for safety against numerical issues.
    nodes, weights = sp.roots_laguerre(50)
    
    def objective(gaps):
        if time.time() - start_time > timeout:
            return 1e10
        roots = np.cumsum(gaps)
        # Calculate P(x)^2 at nodes
        # P(x) = prod (x - r_i)^2
        # log P(x) = 2 * sum log(x - r_i)
        # But we need P(x)^2 = exp(4 * sum log(x - r_i))
        # Direct calculation might overflow.
        # Let's compute log_val = 4 * sum(log(nodes - roots))
        # Then sum(weights * exp(log_val))
        # To avoid overflow, we can shift.
        # Or just compute P(nodes) directly if roots are not too large.
        # Roots are ~ 1-50. Nodes are ~ 0-100.
        # P(x) is product of 12 terms squared.
        # (50-1)^2 * ... might be large.
        # Let's use log space.
        
        # nodes - roots might be negative?
        # Roots are positive. Nodes are positive.
        # If a node is less than a root, log is invalid.
        # P(x) is real. P(x)^2 is positive.
        # We can compute P(x) = prod (x-r_i)^2.
        # Let's compute the product directly in chunks or use log.
        # Since P(x)^2 is always positive, we can compute:
        # val = 0
        # for r in roots:
        #   val += 2 * np.log(np.abs(nodes - r))
        # P_sq = np.exp(val)
        
        diffs = nodes[:, None] - roots[None, :]
        # P(x) = prod (x-r)^2
        # log P_sq = 4 * sum log |x-r|
        log_P_sq = 4.0 * np.sum(np.log(np.abs(diffs)), axis=1)
        
        # Integrate
        # Integral = sum w_j exp(log_P_sq_j)
        # To avoid overflow in exp, find max_log
        max_log = np.max(log_P_sq)
        if np.isinf(max_log):
             return 1e10
        integral = np.sum(weights * np.exp(log_P_sq - max_log)) * np.exp(max_log)
        
        if integral <= 0 or not np.isfinite(integral):
            return 1e10
            
        return -np.log(integral)

    # Initial guess: gaps of 4.0
    x0 = np.full(12, 4.0)
    
    # Bounds for gaps
    # Min gap 0.1, Max gap 10.0
    bounds = [(0.1, 10.0) for _ in range(12)]
    
    best_result = None
    best_score = 1e10
    
    # Differential Evolution
    try:
        res = so.differential_evolution(
            objective, 
            bounds, 
            seed=42, 
            maxiter=200, 
            popsize=15, 
            tol=1e-6,
            update_interval=10
        )
        if res.success or res.fun < best_score:
            best_result = res.x
            best_score = res.fun
    except Exception as e:
        pass

    # Polish with L-BFGS-B
    if best_result is not None:
        if time.time() - start_time < timeout - 60:
            try:
                res2 = so.minimize(
                    objective, 
                    best_result, 
                    method='L-BFGS-B', 
                    bounds=bounds,
                    options={'maxiter': 1000, 'ftol': 1e-12}
                )
                if res2.fun < best_score:
                    best_result = res2.x
                    best_score = res2.fun
            except Exception:
                pass
    else:
        # Fallback to seed if DE failed
        best_result = x0

    roots = np.cumsum(best_result)
    return roots.tolist()
