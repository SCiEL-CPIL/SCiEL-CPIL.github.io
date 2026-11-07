"""Optimized program for the second autocorrelation inequality task."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 145.0  # safety margin before 150s kill

# EVOLVE-BLOCK-START
def _compute_value(h):
    """Compute the objective value for given step heights."""
    h = np.asarray(h, dtype=np.float64)
    if np.any(h < -1e-12) or not np.all(np.isfinite(h)):
        return -1.0
    h = np.clip(h, 0.0, None)
    g = np.convolve(h, h)
    m = len(g)
    y = np.zeros(m + 2)
    y[1:m+1] = g
    w = 1.0 / (m + 1)
    terms = y[:m+1]**2 + y[:m+1]*y[1:m+2] + y[1:m+2]**2
    l2 = (w / 3.0) * np.sum(terms)
    norm_1 = np.sum(g) / (m + 1)
    norm_inf = np.max(g)
    if norm_1 * norm_inf == 0:
        return -1.0
    return l2 / (norm_1 * norm_inf)

def _negative_value(h):
    """Negative of the objective for minimization."""
    return -_compute_value(h)

def _gradient(h):
    """Numerical gradient using central differences."""
    h = np.clip(h, 0.0, None)
    eps = 1e-7
    grad = np.zeros_like(h)
    for i in range(len(h)):
        h_plus = h.copy()
        h_minus = h.copy()
        h_plus[i] += eps
        h_minus[i] -= eps
        grad[i] = (_negative_value(h_plus) - _negative_value(h_minus)) / (2 * eps)
    return grad

def _hump_init():
    """Create a hump-shaped initial guess."""
    x = np.linspace(-1, 1, N)
    h = np.maximum(0, 1 - np.abs(x))
    if np.max(h) > 0:
        h = h / np.max(h)
    return h

def _ramp_init():
    """Create a ramp-shaped initial guess."""
    x = np.linspace(0, 1, N)
    h = x
    return h

def _alternating_init():
    """Create an alternating initial guess."""
    h = np.zeros(N)
    h[::2] = 1.0
    return h

def _two_peak_init():
    """Create a two-peak initial guess."""
    x = np.linspace(-1, 1, N)
    h = np.maximum(0, 1 - np.abs(x - 0.5)) + np.maximum(0, 1 - np.abs(x + 0.5))
    if np.max(h) > 0:
        h = h / np.max(h)
    return h

def _random_init():
    """Create a random initial guess."""
    h = np.random.uniform(0, 1, N)
    return h

def construct():
    """Main entry point: find 50 nonnegative step heights maximizing the value."""
    start_time = time.time()
    
    # List of initializations to try
    inits = [
        _hump_init(),
        _ramp_init(),
        _alternating_init(),
        _two_peak_init(),
        _random_init(),
        _random_init(),
        _random_init(),
    ]
    
    best_h = np.ones(N)
    best_val = _compute_value(best_h)
    
    bounds = [(0.0, None)] * N
    
    for h0 in inits:
        if time.time() - start_time > DEADLINE:
            break
        
        try:
            # Try L-BFGS-B with numerical gradient
            result = minimize(
                _negative_value,
                h0,
                method='L-BFGS-B',
                jac=_gradient,
                bounds=bounds,
                options={
                    'ftol': 1e-10,
                    'gtol': 1e-12,
                    'maxiter': 500,
                    'maxfun': 10000
                }
            )
            
            if result.success or result.fun < -best_val:
                h_new = np.clip(result.x, 0.0, None)
                val_new = _compute_value(h_new)
                if val_new > best_val:
                    best_val = val_new
                    best_h = h_new.copy()
                    
        except Exception:
            pass
        
        # Check deadline
        if time.time() - start_time > DEADLINE - 10:
            break
    
    # Final polish with tighter tolerance if time permits
    if time.time() - start_time < DEADLINE - 20:
        try:
            result = minimize(
                _negative_value,
                best_h,
                method='L-BFGS-B',
                jac=_gradient,
                bounds=bounds,
                options={
                    'ftol': 1e-12,
                    'gtol': 1e-14,
                    'maxiter': 1000,
                    'maxfun': 50000
                }
            )
            
            if result.success or result.fun < -best_val:
                h_new = np.clip(result.x, 0.0, None)
                val_new = _compute_value(h_new)
                if val_new > best_val:
                    best_val = val_new
                    best_h = h_new.copy()
                    
        except Exception:
            pass
    
    # Ensure nonnegative and finite
    best_h = np.clip(best_h, 0.0, None)
    if not np.all(np.isfinite(best_h)):
        best_h = np.ones(N)
    
    return list(best_h)
# EVOLVE-BLOCK-END
