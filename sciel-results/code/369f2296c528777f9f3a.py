"""Second autocorrelation inequality: maximize ||f||2^2 / (||f||1 * ||f*f||inf).
N=50 nonnegative heights on equal intervals of [-1/4,1/4].

Strategy:
1. Interpret the ratio with L1-normalized heights (scale-invariant core).
2. The ratio R = ||h||_2^2 / ||h*h||_inf for L1=1 h.
3. A single spike gives R=1. Two-point distributions can give R > 1.
4. Use global search (Differential Evolution style) + local refinement (L-BFGS-B)
   to find better configurations.
"""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 50

def _compute_ratio(h):
    """Compute the ratio for L1-normalized nonnegative h."""
    h = np.asarray(h, dtype=np.float64)
    h = np.maximum(h, 0.0)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    h = h / l1
    l2sq = np.sum(h * h)
    conv = np.convolve(h, h, mode='full')
    conv_inf = np.max(conv)
    if conv_inf < 1e-15:
        return 0.0
    return l2sq / conv_inf

def _negative_ratio(h):
    """Negative ratio for minimization."""
    return -_compute_ratio(h)

def construct():
    deadline = time.time() + 140.0  # Safety margin under 150s wall-clock
    
    best_h = np.ones(N, dtype=np.float64)
    best_r = _compute_ratio(best_h)
    
    # Try single spike
    spike = np.zeros(N)
    spike[0] = 1.0
    r = _compute_ratio(spike)
    if r > best_r:
        best_r = r
        best_h = spike.copy()
    
    # Try two-point distributions with various positions and weights
    for i in range(N):
        for j in range(i + 1, N):
            if time.time() > deadline - 10.0:
                break
            # Sweep weight
            for a in np.linspace(0.01, 0.99, 50):
                h = np.zeros(N)
                h[i] = a
                h[j] = 1.0 - a
                r = _compute_ratio(h)
                if r > best_r:
                    best_r = r
                    best_h = h.copy()
        if time.time() > deadline - 10.0:
            break
    
    # Local refinement using L-BFGS-B on the best configuration
    if time.time() < deadline - 20.0:
        try:
            res = minimize(_negative_ratio, best_h, method='L-BFGS-B',
                          options={'maxiter': 200, 'ftol': 1e-10})
            h_opt = np.maximum(res.x, 0.0)
            r_opt = _compute_ratio(h_opt)
            if r_opt > best_r:
                best_r = r_opt
                best_h = h_opt.copy()
        except Exception:
            pass
    
    # Try a few random restarts with local refinement if time permits
    if time.time() < deadline - 10.0:
        rng = np.random.default_rng(42)
        for _ in range(10):
            if time.time() > deadline - 10.0:
                break
            # Random sparse initialization
            h_init = np.zeros(N)
            k = rng.integers(1, 6)
            indices = rng.choice(N, size=k, replace=False)
            weights = rng.dirichlet(np.ones(k))
            h_init[indices] = weights
            
            try:
                res = minimize(_negative_ratio, h_init, method='L-BFGS-B',
                              options={'maxiter': 100, 'ftol': 1e-10})
                h_opt = np.maximum(res.x, 0.0)
                r_opt = _compute_ratio(h_opt)
                if r_opt > best_r:
                    best_r = r_opt
                    best_h = h_opt.copy()
            except Exception:
                pass
    
    # Ensure valid nonnegative sequence
    best_h = np.maximum(best_h, 0.0)
    if np.sum(best_h) < 1e-15:
        best_h = np.ones(N)
    
    return [float(x) for x in best_h]
