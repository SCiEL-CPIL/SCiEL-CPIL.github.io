import numpy as np
from scipy.optimize import minimize
import time

def _compute_c4_proxy(roots):
    """
    Compute a proxy for the C4 upper bound given 12 double-root locations.
    The Cohn-Goncalves construction for Fourier sign uncertainty with m=12, n=1
    involves a Laguerre-type polynomial. The C4 bound is related to the
    weighted norm of the polynomial.
    
    We use the objective: minimize the maximum of |P(x)| * x * exp(-x/2)
    where P(x) = prod_i (x - r_i)^2, evaluated on a grid.
    This captures the essential trade-off in the uncertainty bound.
    """
    roots = np.asarray(roots, dtype=np.float64)
    # Ensure positive and sorted
    roots = np.maximum(roots, 1e-10)
    
    # Evaluate on a fine grid
    x = np.linspace(0.01, 100.0, 2000)
    
    # Compute P(x) = prod (x - r_i)^2
    # Use log-sum for numerical stability
    log_abs_P = np.zeros_like(x)
    for r in roots:
        log_abs_P += 2.0 * np.log(np.abs(x - r) + 1e-300)
    
    # Weight: x * exp(-x/2) for n=1
    weight = x * np.exp(-x / 2.0)
    
    # Objective: max of |P(x)| * weight
    obj = np.max(np.exp(log_abs_P) * weight)
    
    return obj

def construct():
    deadline = time.time() + 570  # safety margin before 630s kill
    
    # Initial guess: Laguerre-Gauss nodes
    import numpy.polynomial.laguerre as npl
    roots0, _ = npl.laggauss(12)
    roots0 = roots0.tolist()
    
    best_roots = roots0
    best_val = _compute_c4_proxy(roots0)
    
    # Try multiple optimization strategies
    strategies = [
        ('Nelder-Mead', {'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10}),
        ('L-BFGS-B', {'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}),
        ('Powell', {'maxiter': 2000, 'ftol': 1e-10, 'xtol': 1e-8}),
    ]
    
    for method, options in strategies:
        if time.time() > deadline:
            break
        
        try:
            res = minimize(
                _compute_c4_proxy,
                x0=best_roots,
                method=method,
                options=options,
                bounds=[(1e-6, None)] * 12 if method == 'L-BFGS-B' else None
            )
            if res.success or res.fun < best_val:
                candidate = np.maximum(res.x, 1e-10)
                val = _compute_c4_proxy(candidate)
                if val < best_val:
                    best_val = val
                    best_roots = candidate.tolist()
        except Exception:
            pass
    
    # Final refinement with Nelder-Mead
    if time.time() < deadline - 30:
        try:
            res = minimize(
                _compute_c4_proxy,
                x0=best_roots,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            if res.fun < best_val:
                candidate = np.maximum(res.x, 1e-10)
                val = _compute_c4_proxy(candidate)
                if val < best_val:
                    best_val = val
                    best_roots = candidate.tolist()
        except Exception:
            pass
    
    return best_roots
