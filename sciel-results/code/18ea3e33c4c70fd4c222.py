"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

# EVOLVE-BLOCK-START
def _c4_proxy(roots):
    """Proxy for C4 upper bound based on root configuration.
    Uses the product of (x - r_i) evaluated at key points,
    and the spread of roots as a heuristic for the sign-change bound."""
    roots = np.array(roots)
    if np.any(roots <= 0):
        return 1e10
    if np.any(np.diff(roots) <= 0):
        return 1e10
    # Heuristic: minimize the maximum root and keep roots well-spread
    # The C4 bound in Cohn-Goncalves relates to the norm of the polynomial
    # P(x) = prod (x - r_i)^2 for double roots
    # A simpler proxy: minimize sum of r_i^2 (related to L2 norm)
    return float(np.sum(roots ** 2))

def construct():
    start_time = time.time()
    deadline = 580.0  # safety margin before 630s kill
    
    # Start with the seed
    x0 = np.array([4.0 * (i + 1) for i in range(12)], dtype=float)
    
    best_roots = x0.copy()
    best_score = _c4_proxy(x0)
    
    # Try local optimization
    try:
        result = minimize(_c4_proxy, x0, method='Nelder-Mead',
                         options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8})
        if result.success or _c4_proxy(result.x) < best_score:
            candidate = result.x
            if np.all(candidate > 0) and np.all(np.diff(candidate) > 0):
                best_roots = candidate
                best_score = _c4_proxy(candidate)
    except Exception:
        pass
    
    if time.time() - start_time > deadline:
        return best_roots.tolist()
    
    # Try a few different initializations
    inits = [
        np.array([2.0 * (i + 1) for i in range(12)]),
        np.array([1.0 + i * 3.0 for i in range(12)]),
        np.array([4.0 * (i + 1) ** 0.8 for i in range(12)]),
    ]
    
    for init in inits:
        if time.time() - start_time > deadline:
            break
        try:
            res = minimize(_c4_proxy, init, method='Nelder-Mead',
                          options={'maxiter': 300, 'xatol': 1e-6, 'fatol': 1e-8})
            if np.all(res.x > 0) and np.all(np.diff(res.x) > 0):
                if _c4_proxy(res.x) < best_score:
                    best_roots = res.x
                    best_score = _c4_proxy(res.x)
        except Exception:
            pass
    
    return best_roots.tolist()
# EVOLVE-BLOCK-END
