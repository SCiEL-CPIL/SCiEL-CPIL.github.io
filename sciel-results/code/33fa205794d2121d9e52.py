import numpy as np
from scipy.optimize import differential_evolution
import time

N = 600
TIME_LIMIT = 120  # seconds, with safety margin

def autoconv_max_normalized(h):
    """Compute max of autoconvolution normalized by (sum h)^2."""
    h = np.asarray(h, dtype=float)
    s = h.sum()
    if s == 0:
        return float('inf')
    # Discrete autoconvolution
    c = np.correlate(h, h, mode='full')
    max_c = c.max()
    return max_c / (s * s)

def construct():
    start_time = time.time()
    
    # Initial guess: flat
    best_h = np.ones(N)
    best_val = autoconv_max_normalized(best_h)
    
    # Try a few heuristic shapes
    # Shape 1: raised cosine-like (smooth)
    x = np.linspace(-1, 1, N)
    h_cos = np.maximum(0, 1 + 0.5 * np.cos(np.pi * x))
    best_h = np.maximum(h_cos, 0)
    val = autoconv_max_normalized(best_h)
    if val < best_val:
        best_val = val
    
    # Shape 2: flat top with tapered edges
    h_flat = np.ones(N)
    taper = 50
    h_flat[:taper] = np.linspace(0.3, 1, taper)
    h_flat[-taper:] = np.linspace(1, 0.3, taper)
    val = autoconv_max_normalized(h_flat)
    if val < best_val:
        best_val = val
        best_h = h_flat.copy()
    
    # Shape 3: uniform (baseline)
    h_unif = np.ones(N)
    val = autoconv_max_normalized(h_unif)
    if val < best_val:
        best_val = val
        best_h = h_unif.copy()
    
    # Use differential evolution on a reduced set of parameters
    # Parameterize as a low-dimensional spline
    n_knots = 10
    def spline_to_heights(knots):
        """Interpolate knots to N heights, ensuring nonnegativity."""
        x_knots = np.linspace(0, N - 1, n_knots)
        x_full = np.arange(N)
        h = np.interp(x_full, x_knots, knots)
        return np.maximum(h, 0)
    
    def objective(knots):
        h = spline_to_heights(knots)
        return autoconv_max_normalized(h)
    
    # Multi-start differential evolution
    bounds = [(0, 2)] * n_knots
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > TIME_LIMIT - 30:
        return list(best_h)
    
    # Run differential evolution with limited iterations
    n_iter = 30
    popsize = 10
    
    try:
        result = differential_evolution(
            objective, bounds, maxiter=n_iter, popsize=popsize,
            tol=1e-4, seed=42, polish=False
        )
        if result.success or result.fun < best_val:
            h_opt = spline_to_heights(result.x)
            val_opt = autoconv_max_normalized(h_opt)
            if val_opt < best_val:
                best_val = val_opt
                best_h = h_opt
    except Exception:
        pass
    
    # Check time again
    elapsed = time.time() - start_time
    if elapsed > TIME_LIMIT - 15:
        return list(best_h)
    
    # Local refinement using scipy.optimize.minimize
    from scipy.optimize import minimize
    
    def local_objective(knots):
        h = spline_to_heights(knots)
        return autoconv_max_normalized(h)
    
    try:
        res = minimize(
            local_objective, result.x, method='Nelder-Mead',
            options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-6}
        )
        if res.fun < best_val:
            h_opt = spline_to_heights(res.x)
            val_opt = autoconv_max_normalized(h_opt)
            if val_opt < best_val:
                best_val = val_opt
                best_h = h_opt
    except Exception:
        pass
    
    # Final check
    elapsed = time.time() - start_time
    if elapsed > TIME_LIMIT - 5:
        return list(best_h)
    
    return list(best_h)
