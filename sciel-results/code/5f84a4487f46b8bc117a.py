import numpy as np
from scipy.special import roots_laguerre
from scipy.optimize import minimize
import time

def _c4_bound(roots):
    """Compute the C4 upper bound from root locations."""
    roots = np.asarray(roots, dtype=float)
    if np.any(roots <= 0):
        return 1e10
    # C = prod_{i=1}^m (pi/x_i + x_i/pi)
    log_c = np.sum(np.log(np.pi / roots + roots / np.pi))
    return np.exp(log_c)

def construct():
    start_time = time.time()
    deadline = start_time + 570  # safety margin before 600s limit
    
    # Initial guess: zeros of Laguerre polynomial L_12
    r = roots_laguerre(12)
    x0 = np.sort(r[0].copy())
    
    best_roots = x0.copy()
    best_score = _c4_bound(best_roots)
    
    # Objective function
    def objective(x):
        if np.any(x <= 0):
            return 1e10
        return _c4_bound(x)
    
    # Multi-start optimization
    np.random.seed(42)
    n_starts = 20
    for i in range(n_starts):
        if time.time() > deadline:
            break
        
        if i == 0:
            x_start = x0.copy()
        elif i == 1:
            # All near pi, spread out
            x_start = np.linspace(2.8, 3.5, 12)
        elif i == 2:
            # Geometric spacing near pi
            x_start = np.geomspace(2.0, 5.0, 12)
        else:
            # Perturb Laguerre zeros
            scale = 0.02 * (i - 2)
            x_start = x0 * (1 + np.random.randn(12) * scale)
            x_start = np.sort(np.maximum(x_start, 0.5))
        
        try:
            result = minimize(objective, x_start, method='L-BFGS-B',
                             bounds=[(0.5, 50)] * 12,
                             options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-15})
            
            if result.success and result.fun < best_score:
                best_score = result.fun
                best_roots = result.x.copy()
        except Exception:
            pass
    
    return best_roots.tolist()
