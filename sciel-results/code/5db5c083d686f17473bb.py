import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

def _compute_c4_bound(roots):
    """
    Compute the C4 upper bound for the Cohn-Goncalves construction.
    The bound is related to the weighted L-infinity norm of the polynomial
    P(x) = prod (x - r_i)^2 with weight w(x) = x * exp(-x/2).
    We use a non-uniform grid concentrated where the Laguerre polynomial
    typically has significant variation (near the origin and the turning point).
    """
    roots = np.asarray(roots, dtype=np.float64)
    # Ensure positive
    roots = np.maximum(roots, 1e-12)
    
    # Generate a smart grid:
    # Laguerre zeros for m=12 are roughly in [0, 4m] = [0, 48].
    # The weight x*exp(-x/2) peaks at x=2.
    # The polynomial prod(x-r_i)^2 grows large for large x.
    # To capture the max of |P(x)| * w(x), we need a grid that resolves
    # the oscillations of P(x) and the decay of w(x).
    
    # Use a transformation to concentrate points near the origin and spread them out.
    # Let u = sqrt(x), so x = u^2.
    # Zeros of L_m(x) scale like m. For m=12, zeros are ~ 1, 4, 8, ...
    # Let's use a grid that is uniform in sqrt(x) up to a certain limit, then linear.
    
    # Max root is likely around 40-50.
    # Let's evaluate up to 100 to be safe, but concentrate below 50.
    
    # Grid 1: Uniform in sqrt(x) from 0 to 10 (covers x=0 to 100)
    u = np.linspace(0.01, 10.0, 4000)
    x = u**2
    
    # Compute log |P(x)|
    # P(x) = prod (x - r_i)^2
    # log |P(x)| = 2 * sum log |x - r_i|
    
    log_abs_P = np.zeros_like(x)
    for r in roots:
        # Avoid log(0)
        diff = x - r
        abs_diff = np.abs(diff)
        # Where diff is very small, log is -inf, but P(x) is 0.
        # We want to find the max, so zeros are not the max.
        # However, numerically, we should handle it.
        # If x == r, term is 0.
        # We can add a small epsilon to avoid -inf in log, but then exp(log) will be small.
        # Actually, if any factor is 0, P(x) is 0.
        # The max will not be at the roots.
        # So we can safely ignore the exact zero points or clip.
        log_abs_P += 2.0 * np.log(abs_diff + 1e-300)
        
    # Weight
    weight = x * np.exp(-x / 2.0)
    
    # Objective: max |P(x)| * weight
    # We compute in log space to avoid overflow/underflow
    log_obj = log_abs_P + np.log(weight + 1e-300)
    
    # The max of the log is the log of the max
    max_log_val = np.max(log_obj)
    
    # Return the actual value
    return np.exp(max_log_val)

def construct():
    deadline = time.time() + 570  # Safety margin
    
    # Initial guess: Laguerre-Gauss nodes
    import numpy.polynomial.laguerre as npl
    roots0, _ = npl.laggauss(12)
    roots0 = roots0.tolist()
    
    best_roots = roots0
    best_val = _compute_c4_bound(roots0)
    
    # Strategy 1: Differential Evolution for global search
    # Bounds: Laguerre zeros for m=12 are between ~1 and ~45.
    # Let's allow a wide range [0.1, 100]
    bounds = [(0.1, 100.0)] * 12
    
    if time.time() < deadline - 120:
        try:
            # DE is stochastic, so we need to be careful with time.
            # maxiter=100, popsize=15 -> 1500 evaluations.
            # Each eval is ~4000 points * 12 roots -> fast.
            res_de = differential_evolution(
                _compute_c4_bound,
                bounds=bounds,
                maxiter=100,
                popsize=15,
                tol=1e-8,
                seed=42,
                polish=False  # We will polish later
            )
            if res_de.fun < best_val:
                best_val = res_de.fun
                best_roots = res_de.x.tolist()
        except Exception:
            pass
    
    # Strategy 2: Local refinement with L-BFGS-B
    # Start from the best of initial and DE result
    current_roots = np.array(best_roots)
    
    if time.time() < deadline - 60:
        try:
            res_lbfgs = minimize(
                _compute_c4_bound,
                x0=current_roots,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if res_lbfgs.fun < best_val:
                best_val = res_lbfgs.fun
                best_roots = res_lbfgs.x.tolist()
        except Exception:
            pass
            
    # Strategy 3: Final fine-tuning with Nelder-Mead if time permits
    if time.time() < deadline - 10:
        try:
            res_nm = minimize(
                _compute_c4_bound,
                x0=np.array(best_roots),
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            if res_nm.fun < best_val:
                best_val = res_nm.fun
                best_roots = res_nm.x.tolist()
        except Exception:
            pass
            
    # Ensure final result is valid (positive)
    final_roots = np.maximum(np.array(best_roots), 1e-10).tolist()
    
    return final_roots
