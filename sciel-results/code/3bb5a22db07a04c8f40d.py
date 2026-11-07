"""Optimized program for the second autocorrelation inequality task."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 50

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

def _hump_init():
    """Create a hump-shaped initial guess."""
    x = np.linspace(-1, 1, N)
    h = np.maximum(0, 1 - np.abs(x))
    h = h / np.max(h)
    return h

def _triangle_init():
    """Create a triangular initial guess."""
    x = np.linspace(-1, 1, N)
    h = np.maximum(0, 1 - 2*np.abs(x))
    if np.max(h) > 0:
        h = h / np.max(h)
    return h

def _two_peak_init():
    """Create a two-peak initial guess."""
    x = np.linspace(-1, 1, N)
    h1 = np.exp(-20*(x - 0.4)**2)
    h2 = np.exp(-20*(x + 0.4)**2)
    h = h1 + h2
    h = h / np.max(h)
    return h

def _ramp_init():
    """Create a ramp initial guess."""
    h = np.linspace(0, 1, N)
    return h

def _alternating_init():
    """Create an alternating initial guess."""
    h = np.ones(N)
    h[::2] = 0.5
    return h

def _optimize_from_start(h0, deadline):
    """Optimize from a given starting point using L-BFGS-B."""
    start_time = time.time()
    
    # First, do a local optimization in log-space for positivity
    h0 = np.maximum(h0, 1e-6)
    log_h0 = np.log(h0)
    
    def neg_value_log(log_h):
        if time.time() > deadline:
            return 1e10
        h = np.exp(log_h)
        # Scale to keep values reasonable
        h = h / np.max(h) if np.max(h) > 0 else h
        return -_compute_value(h)
    
    # Optimize in log-space
    result_log = minimize(neg_value_log, log_h0, method='L-BFGS-B',
                         options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12})
    
    if time.time() > deadline:
        h = np.exp(log_h0)
        h = h / np.max(h) if np.max(h) > 0 else h
        return h, _compute_value(h)
    
    h = np.exp(result_log.x)
    h = h / np.max(h) if np.max(h) > 0 else h
    
    # Then, do direct-space optimization with bounds
    def neg_value_direct(h):
        if time.time() > deadline:
            return 1e10
        h = np.clip(h, 0, None)
        return -_compute_value(h)
    
    bounds = [(0, None)] * N
    result_direct = minimize(neg_value_direct, h, method='L-BFGS-B',
                            bounds=bounds, options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12})
    
    if time.time() > deadline:
        return h, _compute_value(h)
    
    h_final = np.clip(result_direct.x, 0, None)
    val = _compute_value(h_final)
    return h_final, val

def construct():
    """Main construct function that searches for optimal step heights."""
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin before 150s limit
    
    # Generate diverse initial guesses
    inits = [
        _hump_init(),
        _triangle_init(),
        _two_peak_init(),
        _ramp_init(),
        _alternating_init(),
        np.ones(N),
        np.concatenate([np.ones(N//2), np.zeros(N - N//2)]),
        np.concatenate([np.zeros(N//2), np.ones(N - N//2)]),
    ]
    
    best_h = np.ones(N)
    best_val = _compute_value(best_h)
    
    # Try each initial guess
    for i, h0 in enumerate(inits):
        if time.time() > deadline - 5.0:
            break
        
        h_opt, val = _optimize_from_start(h0, deadline)
        
        if val > best_val:
            best_val = val
            best_h = h_opt
    
    # If we have time, try differential evolution for global exploration
    if time.time() < deadline - 20.0:
        def neg_value_de(h):
            if time.time() > deadline:
                return 1e10
            h = np.clip(h, 0, None)
            return -_compute_value(h)
        
        bounds = [(0, 1)] * N
        try:
            result_de = differential_evolution(neg_value_de, bounds, 
                                              maxiter=30, popsize=15,
                                              seed=42, tol=1e-10,
                                              polish=True)
            if time.time() < deadline - 5.0:
                h_de = np.clip(result_de.x, 0, None)
                val_de = _compute_value(h_de)
                if val_de > best_val:
                    best_val = val_de
                    best_h = h_de
        except Exception:
            pass
    
    # Final polish with more iterations if we have time
    if time.time() < deadline - 3.0 and best_val > 0:
        def neg_value_final(h):
            if time.time() > deadline:
                return 1e10
            h = np.clip(h, 0, None)
            return -_compute_value(h)
        
        bounds = [(0, None)] * N
        try:
            result_final = minimize(neg_value_final, best_h, method='L-BFGS-B',
                                   bounds=bounds, options={'maxiter': 1000, 
                                                          'ftol': 1e-15, 'gtol': 1e-13})
            if time.time() < deadline:
                h_final = np.clip(result_final.x, 0, None)
                val_final = _compute_value(h_final)
                if val_final > best_val:
                    best_val = val_final
                    best_h = h_final
        except Exception:
            pass
    
    # Ensure nonnegative and finite
    best_h = np.clip(best_h, 0, None)
    best_h = np.where(np.isfinite(best_h), best_h, 0.0)
    
    return list(best_h)
# EVOLVE-BLOCK-END
