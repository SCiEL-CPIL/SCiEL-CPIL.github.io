"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import genlaguerre, roots_laguerre

# EVOLVE-BLOCK-START
def _c4_bound(roots, m=12, n=1):
    """
    Compute the C4 upper bound for the Cohn-Goncalves Laguerre construction.
    The bound is based on the product of Laguerre polynomial evaluations
    at the double-root locations.
    """
    roots = np.asarray(roots, dtype=float)
    if len(roots) != m:
        return 1e10
    if np.any(roots <= 0):
        return 1e10
    if not np.all(np.diff(roots) > 0):
        return 1e10
    
    # Compute Laguerre polynomial L_m(x)
    L = genlaguerre(m, n)
    
    # Evaluate at roots
    vals = L(roots)
    
    # The C4 bound involves the product of inverse squared Laguerre values
    # and the roots themselves
    # C4 = prod(x_i) / prod(|L_m(x_i)|^2)
    # But this could be very large. Let's try:
    # C4 = prod(1/(x_i * |L_m(x_i)|^2))
    
    # Try a log-sum formulation for numerical stability
    log_obj = np.sum(np.log(roots)) - 2.0 * np.sum(np.log(np.abs(vals) + 1e-300))
    
    if not np.isfinite(log_obj):
        return 1e10
    
    obj = np.exp(log_obj)
    
    return obj

def construct():
    start_time = time.time()
    deadline = 600.0
    safety_margin = 30.0
    
    # Start with the seed configuration
    best_roots = np.array([4.0 * (i + 1) for i in range(12)], dtype=float)
    best_obj = _c4_bound(best_roots)
    
    elapsed = time.time() - start_time
    if elapsed + safety_margin > deadline:
        return [float(x) for x in best_roots]
    
    # Try multiple optimization strategies
    # Strategy 1: Optimize with log-spaced initial guesses
    for restart in range(30):
        elapsed = time.time() - start_time
        if elapsed + safety_margin > deadline:
            break
        
        if restart == 0:
            x0 = best_roots.copy()
        elif restart % 3 == 1:
            # Geometric spacing
            ratio = np.random.uniform(0.5, 1.5)
            x0 = 0.5 * ratio ** np.arange(12)
            x0 = np.sort(x0)
        elif restart % 3 == 2:
            # Power law
            alpha = np.random.uniform(0.3, 2.5)
            x0 = 0.5 * np.arange(1, 13) ** alpha
            x0 = np.sort(x0)
        else:
            # Perturbation of best
            x0 = best_roots * np.exp(np.random.randn(12) * 0.3)
            x0 = np.sort(x0)
            x0 = np.maximum(x0, 0.01)
        
        def objective(params):
            elapsed_inner = time.time() - start_time
            if elapsed_inner + safety_margin > deadline:
                return 1e10
            return _c4_bound(params)
        
        try:
            result = minimize(objective, x0, method='L-BFGS-B',
                             bounds=[(0.01, 1000.0)] * 12,
                             options={'maxiter': 200, 'ftol': 1e-12})
            if result.fun < best_obj and np.isfinite(result.fun):
                best_obj = result.fun
                best_roots = result.x.copy()
        except Exception:
            pass
    
    # Ensure validity
    best_roots = np.sort(best_roots)
    best_roots = np.maximum(best_roots, 0.001)
    
    return [float(x) for x in best_roots]
# EVOLVE-BLOCK-END
