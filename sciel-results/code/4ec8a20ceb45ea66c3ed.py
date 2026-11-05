"""Optimized program for the third autocorrelation inequality task."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _objective(h):
    """Compute the value to minimize."""
    conv = np.convolve(h, h)
    max_conv = np.max(conv)
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e10
    return abs(2.0 * N * max_conv / (s * s))

def _construct_optimized():
    """Optimize h to minimize the autocorrelation ratio."""
    deadline = time.time() + 100  # 100s budget, leave 20s margin
    
    # Initial guess: ones
    h0 = np.ones(N)
    
    # We can scale freely, so fix sum to 1 for optimization
    # But the objective is scale-invariant, so let's just optimize directly
    # Use a random initialization with some structure
    
    best_h = h0.copy()
    best_val = _objective(best_h)
    
    # Try several random initializations with different structures
    rng = np.random.default_rng(42)
    
    initializations = [
        np.ones(N),
        rng.standard_normal(N),
        np.sign(np.cos(2 * np.pi * np.arange(N) / N)),
        np.sin(2 * np.pi * np.arange(N) / N) + 0.5,
        np.ones(N) * np.exp(-((np.arange(N) - N/2)**2) / (2 * (N/4)**2)),
    ]
    
    for h_init in initializations:
        if time.time() > deadline - 30:
            break
        
        h_init = h_init.astype(float)
        # Ensure sum is not zero
        if abs(np.sum(h_init)) < 1e-10:
            h_init[0] += 1.0
        
        try:
            result = minimize(
                _objective,
                h_init,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_h = result.x.copy()
        except Exception:
            pass
    
    # Local refinement with more iterations on the best
    if time.time() < deadline - 10:
        try:
            result = minimize(
                _objective,
                best_h,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_h = result.x.copy()
        except Exception:
            pass
    
    # Final check: ensure validity
    final_val = _objective(best_h)
    if not np.isfinite(final_val) or abs(np.sum(best_h)) < 1e-12:
        return np.ones(N)
    
    return best_h

def construct():
    """Construct step heights minimizing the autocorrelation ratio."""
    try:
        return _construct_optimized()
    except Exception:
        return np.ones(N)
# EVOLVE-BLOCK-END
