import numpy as np
from scipy.optimize import minimize
import time

N = 600
w = 1.0 / (2 * N)  # bin width = (1/2) / 600

def autoconv_max(h):
    """Compute max of autoconvolution / (sum h)^2 for step heights h."""
    s = np.sum(h)
    if s == 0:
        return float('inf')
    conv = np.convolve(h, h, mode='full')
    return np.max(conv) / (s * s)

def make_double_box(inner_frac, h_inner, h_outer):
    """Create a symmetric double-box: outer portions have height h_outer,
    inner portion has height h_inner. inner_frac is fraction of total width
    occupied by the inner (low) region, centered."""
    h = np.empty(N)
    # inner region: centered, width = inner_frac * N bins
    inner_bins = int(round(inner_frac * N / 2))
    start = N // 2 - inner_bins
    end = N // 2 + inner_bins
    h[:] = h_outer
    h[start:end] = h_inner
    return h

def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # safety margin
    
    best_score = autoconv_max(np.ones(N))  # seed baseline
    best_h = np.ones(N)
    
    # Grid search over double-box parameters
    # inner_frac: fraction of total width that is the inner (low) region
    # h_ratio: h_inner / h_outer
    # We fix h_outer = 1, vary h_inner in [0, 1]
    
    inner_fracs = np.linspace(0.0, 0.9, 20)
    h_ratios = np.linspace(0.0, 1.0, 20)
    
    for ifr in inner_fracs:
        for hr in h_ratios:
            if time.time() > deadline:
                break
            h = make_double_box(ifr, hr, 1.0)
            if np.all(h < 0):
                continue
            sc = autoconv_max(h)
            if sc < best_score:
                best_score = sc
                best_h = h.copy()
    
    # Local polish with scipy
    if time.time() < deadline - 20:
        def objective(x):
            inner_frac, h_inner = x
            h_inner = max(0.0, h_inner)
            inner_frac = min(max(inner_frac, 0.0), 0.95)
            h = make_double_box(inner_frac, h_inner, 1.0)
            return autoconv_max(h)
        
        x0 = [best_h_inner_frac, best_h_inner_val] if hasattr(best_h, '_meta') else [0.3, 0.1]
        # Extract initial params from best_h
        # For now, just try a few starting points
        for x0 in [[0.2, 0.0], [0.3, 0.1], [0.5, 0.0], [0.4, 0.2]]:
            if time.time() > deadline - 10:
                break
            try:
                res = minimize(objective, x0, method='Nelder-Mead', 
                              options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8})
                if res.fun < best_score:
                    best_score = res.fun
                    h = make_double_box(res.x[0], max(0, res.x[1]), 1.0)
                    best_h = h
            except:
                pass
    
    return list(best_h)
