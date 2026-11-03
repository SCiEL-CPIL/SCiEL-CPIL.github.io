"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def _condition_number_proxy(roots):
    """
    Proxy for the uncertainty bound: minimize the condition number
    of the Vandermonde matrix constructed from the roots.
    This encourages roots to be well-separated and not clustered,
    which aligns with the observation that uniform roots outperform
    clustered Laguerre roots.
    """
    x = np.array(roots)
    # Construct Vandermonde matrix V where V[i, j] = x[i]^j
    # We use a subset of powers to keep it small, e.g., up to degree 11
    V = np.vander(x, N=12, increasing=True)
    # Calculate condition number
    cond = np.linalg.cond(V)
    return cond

def construct():
    m = 12
    n = 1
    
    # Initial guess: Uniform roots similar to seed (4, 8, ..., 48)
    # But we will optimize the positions.
    # We constrain roots to be positive and sorted.
    
    # Parameterization: r_i = offset + step * i
    # Or free parameters with sorting.
    # Let's use free parameters and sort them at the end.
    # To ensure positivity, we use softplus or just bounds.
    
    # Initial guess from seed
    x0 = np.array([4.0 * (i + 1) for i in range(12)])
    
    # We want to minimize the condition number proxy.
    # Using Nelder-Mead as it doesn't require gradients.
    
    def objective(params):
        # Ensure positivity and sorting
        roots = np.sort(np.abs(params))
        if np.any(roots < 1e-6):
            return 1e10
        return _condition_number_proxy(roots)
    
    try:
        # Optimize
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'maxiter': 2000, 'xatol': 1e-5, 'fatol': 1e-5})
        
        # Extract optimized roots
        opt_roots = np.sort(np.abs(res.x))
        
        # Ensure they are strictly positive
        if np.any(opt_roots <= 0):
            opt_roots = x0
        else:
            # If the optimization converged to a very tight cluster (bad cond num might be high? No, tight cluster -> high cond num).
            # So minimizing cond num should spread them out.
            pass
            
    except Exception:
        opt_roots = x0

    return list(opt_roots)
# EVOLVE-BLOCK-END
