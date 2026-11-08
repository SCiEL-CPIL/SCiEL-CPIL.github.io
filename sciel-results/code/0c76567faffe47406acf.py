"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.signal import fftconvolve

N = 50

# EVOLVE-BLOCK-START
def _compute_ratio(h):
    """Compute ||h||_2^2 / (||h||_1 * ||h*h||_inf) for a step function."""
    h = np.asarray(h, dtype=np.float64)
    if np.any(h < 0):
        return 0.0
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h**2)
    # Autoconvolution (full)
    hh = fftconvolve(h, h, mode='full')
    hh_inf = np.max(np.abs(hh))
    if hh_inf == 0:
        return 0.0
    return l2_sq / (l1 * hh_inf)

def _negative_ratio(h):
    return -_compute_ratio(h)

def construct():
    deadline = time.time() + 110.0  # safety margin before 120s limit
    
    # Start with the 2-point 2:1 cluster (known to give ~1.25 ratio)
    # h = [2/3, 1/3, 0, ..., 0] normalized to L1=1
    best_h = np.zeros(N)
    best_h[0] = 2.0/3.0
    best_h[1] = 1.0/3.0
    best_score = _compute_ratio(best_h)
    
    # Also try other simple configurations
    candidates = [
        np.zeros(N),  # uniform-ish
        np.ones(N)/N,  # uniform
    ]
    for c in candidates:
        s = _compute_ratio(c)
        if s > best_score:
            best_score = s
            best_h = c.copy()
    
    # Try random restarts with local optimization
    np.random.seed(42)
    n_restarts = 50
    
    for r in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Random initial point
        x0 = np.random.dirichlet(np.ones(N) * 0.5)  # random nonnegative, sums to 1
        
        # Optimize
        try:
            result = minimize(_negative_ratio, x0, method='Nelder-Mead',
                            options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10})
            if result.success or result.fun < -best_score:
                h_opt = np.maximum(result.x, 0)  # ensure nonnegative
                if np.sum(h_opt) > 0:
                    h_opt = h_opt / np.sum(h_opt)  # normalize
                    s = _compute_ratio(h_opt)
                    if s > best_score:
                        best_score = s
                        best_h = h_opt.copy()
        except Exception:
            pass
    
    # Convert to list
    return list(best_h)
# EVOLVE-BLOCK-END
