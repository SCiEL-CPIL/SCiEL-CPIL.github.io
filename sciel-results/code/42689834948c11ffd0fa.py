"""Second autocorrelation inequality - optimized."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50

def _compute_value(h):
    h = np.asarray(h, dtype=float)
    g = np.convolve(h, h)
    m = len(g)
    y = np.concatenate(([0.0], g, [0.0]))
    w = 1.0 / (m + 1)
    # l2 = sum over i in 0..m of (w/3)*(y[i]^2 + y[i]*y[i+1] + y[i+1]^2)
    y_sq = y[:m+1]**2 + y[:m+1]*y[1:m+2] + y[1:m+2]**2
    l2 = (w / 3.0) * np.sum(y_sq)
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    if norm_1 * norm_inf < 1e-300:
        return 0.0
    return l2 / (norm_1 * norm_inf)

def _neg_value_from_params(params):
    """params: raw values, h = softplus(params), then normalize to sum=1"""
    h = np.exp(params) / (np.exp(params).sum() + 1e-300)
    return -_compute_value(h)

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 130.0  # safety margin before 150s kill
    
    best_h = np.ones(N)
    best_val = _compute_value(best_h)
    
    # Try several initial shapes and optimize
    rng = np.random.default_rng(42)
    
    # Initial candidates: uniform, Gaussian-like, triangular, etc.
    candidates = []
    # Uniform
    candidates.append(np.ones(N))
    # Gaussian centered
    x = np.linspace(-1, 1, N)
    candidates.append(np.exp(-2*x**2))
    # Triangular
    candidates.append(np.maximum(0, 1 - 2*np.abs(np.linspace(-0.5, 0.5, N))))
    # Cosine bump
    candidates.append(np.maximum(0, np.cos(np.pi * np.linspace(-0.5, 0.5, N))))
    # A few random
    for _ in range(5):
        candidates.append(np.abs(rng.standard_normal(N)) + 0.1)
    
    for h0 in candidates:
        if time.time() > deadline - 10:
            break
        h0 = np.asarray(h0, dtype=float)
        if np.any(h0 < 0):
            h0 = np.abs(h0)
        h0 = h0 / h0.sum()
        
        # Parameterize with log for positivity
        params0 = np.log(h0 + 1e-15)
        
        try:
            res = minimize(_neg_value_from_params, params0, method='Nelder-Mead',
                          options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8,
                                   'adaptive': True})
            if res.fun < -best_val:
                h_opt = np.exp(res.x) / (np.exp(res.x).sum() + 1e-300)
                best_val = -res.fun
                best_h = h_opt
        except Exception:
            pass
    
    # Try a few more random restarts if time permits
    for _ in range(10):
        if time.time() > deadline - 5:
            break
        h0 = np.abs(rng.standard_normal(N)) + 0.01
        h0 = h0 / h0.sum()
        params0 = np.log(h0 + 1e-15)
        try:
            res = minimize(_neg_value_from_params, params0, method='Nelder-Mead',
                          options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8,
                                   'adaptive': True})
            if res.fun < -best_val:
                h_opt = np.exp(res.x) / (np.exp(res.x).sum() + 1e-300)
                best_val = -res.fun
                best_h = h_opt
        except Exception:
            pass
    
    # Scale back to reasonable values (scale-invariant, but keep finite)
    # The value is scale-invariant, so any positive scale works.
    # Use sum(h) = 1 for cleanliness
    best_h = best_h / best_h.sum()
    
    return list(best_h)
# EVOLVE-BLOCK-END
