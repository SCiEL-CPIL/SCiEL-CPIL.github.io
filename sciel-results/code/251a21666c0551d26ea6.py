import numpy as np
import time

N = 50
W = 0.5 / N  # width of each step = 0.01

def autoconv_peak(h):
    """Compute supremum norm of the autoconvolution of the step function."""
    # h is array of N nonnegative heights, each on interval of width W
    # f(s) = h[i] for s in [i*W - 0.25, (i+1)*W - 0.25]
    # (f*f)(t) = integral f(s) f(t-s) ds
    # We can compute this on a fine grid or analytically.
    # The autoconv of a step function is piecewise linear.
    # Peak occurs at some t. Let's compute on a grid of lags.
    # The relevant lags are multiples of W, but peak could be between.
    # Actually for step functions, the autoconv is piecewise linear with kinks at multiples of W.
    # The maximum is at one of the kinks or we can sample.
    
    # Discrete autoconv: (f*f)(t) for t = k*W
    # (f*f)(k*W) = sum_i h[i] * h[i-k] * W  (with h out of range = 0)
    # But this is the value at the grid points. The actual peak might be slightly different
    # but for step functions the autoconv is linear between grid points, so max is at grid points.
    
    # Actually, the autoconvolution of two step functions: at lag t = m*W, 
    # (f*f)(m*W) = W * sum_i h[i] h[i-m]
    # The function is piecewise linear and the max over t is the max over these values
    # (since it's linear between, max at endpoints).
    
    h = np.asarray(h, dtype=float)
    n = len(h)
    # Compute all lags
    max_val = 0.0
    for m in range(-n+1, n):
        # sum_i h[i] * h[i-m]
        if m >= 0:
            # i from m to n-1
            s = np.dot(h[m:], h[:n-m])
        else:
            mm = -m
            s = np.dot(h[mm:], h[:n-mm])
        val = s * W
        if val > max_val:
            max_val = val
    return max_val

def compute_ratio(h):
    h = np.asarray(h, dtype=float)
    h = np.maximum(h, 0.0)
    l2_sq = np.sum(h**2) * W
    l1 = np.sum(h) * W
    if l1 == 0:
        return 0.0
    peak = autoconv_peak(h)
    if peak == 0:
        return 0.0
    return l2_sq / (l1 * peak)

def construct():
    deadline = time.time() + 140.0  # safety margin
    
    best_h = np.ones(N)
    best_val = compute_ratio(best_h)
    
    # Try a few initializations
    inits = []
    # Uniform
    inits.append(np.ones(N))
    # Single spike
    for pos in range(N):
        v = np.zeros(N)
        v[pos] = 1.0
        inits.append(v)
    # Gaussian-like
    x = np.linspace(-1, 1, N)
    for sigma in [0.2, 0.5, 1.0]:
        g = np.exp(-x**2 / (2*sigma**2))
        inits.append(g)
    # Two spikes
    for gap in [1, 5, 10, 20]:
        v = np.zeros(N)
        v[0] = 1.0
        if gap < N:
            v[gap] = 1.0
        inits.append(v)
    
    # Evaluate inits
    for v in inits:
        val = compute_ratio(v)
        if val > best_val:
            best_val = val
            best_h = v.copy()
    
    # Local optimization: coordinate ascent / Nelder-Mead lite
    # Use a simple derivative-free local search
    from scipy.optimize import minimize
    
    def neg_ratio(x):
        x = np.maximum(x, 0.0)
        return -compute_ratio(x)
    
    # Try Nelder-Mead from best init
    if time.time() < deadline - 5:
        try:
            res = minimize(neg_ratio, best_h, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8})
            h_opt = np.maximum(res.x, 0.0)
            val_opt = compute_ratio(h_opt)
            if val_opt > best_val:
                best_val = val_opt
                best_h = h_opt
        except Exception:
            pass
    
    # Try from a few other inits if time
    for v in inits[:10]:
        if time.time() >= deadline - 5:
            break
        try:
            res = minimize(neg_ratio, v, method='Nelder-Mead',
                          options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-7})
            h_opt = np.maximum(res.x, 0.0)
            val_opt = compute_ratio(h_opt)
            if val_opt > best_val:
                best_val = val_opt
                best_h = h_opt
        except Exception:
            pass
    
    return list(best_h)
