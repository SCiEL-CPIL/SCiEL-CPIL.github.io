"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 570.0  # safety margin before 600s limit

    m = 12  # number of double roots
    n = 1   # dimension

    # Use the roots of the generalized Laguerre polynomial L_12^(0)(x)
    # as the initial guess. These are the nodes for Gauss-Laguerre quadrature
    # and provide a physically consistent baseline for the Cohn-Goncalves construction.
    # scipy.special.roots_laguerre(n) returns the roots of L_n(x) (alpha=0)
    from scipy.special import roots_laguerre
    laguerre_roots, _ = roots_laguerre(m)
    x0 = np.array(laguerre_roots)
    
    # Ensure positivity (Laguerre roots are positive, but just in case)
    x0 = np.maximum(x0, 1e-8)
    
    # Sort to ensure ordering
    x0 = np.sort(x0)
    
    # Objective: minimize the C4 upper bound.
    # For the Cohn-Goncalves construction with Laguerre polynomials,
    # the constant C4 is related to the product of the double roots.
    # Minimizing the product (via log-sum) is the natural objective.
    def objective(x):
        if np.any(x <= 0):
            return 1e10
        return np.sum(np.log(x))
    
    # Bounds: roots must be positive
    bounds = [(1e-8, None)] * m
    
    # Constraints: ordering x[i] < x[i+1]
    cons = []
    for i in range(m - 1):
        cons.append({
            'type': 'ineq',
            'fun': lambda x, i=i: x[i+1] - x[i] - 1e-8
        })
    
    # Upper bound to prevent unbounded solutions
    for i in range(m):
        cons.append({
            'type': 'ineq',
            'fun': lambda x, i=i: 10000.0 - x[i]
        })
    
    # Run optimization starting from Laguerre roots
    best_x = x0.copy()
    best_val = objective(x0)
    
    try:
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 1000, 'ftol': 1e-14}
        )
        
        elapsed = time.time() - start_time
        if elapsed < deadline:
            if result.success and result.fun < best_val:
                best_x = result.x
                best_val = result.fun
    except Exception:
        pass
    
    # Ensure roots are positive and ordered
    best_x = np.maximum(best_x, 1e-8)
    for i in range(1, m):
        if best_x[i] <= best_x[i-1]:
            best_x[i] = best_x[i-1] + 1e-8
    
    # Convert to list of floats
    return [float(x) for x in best_x]
# EVOLVE-BLOCK-END
