"""Optimized program for the second autocorrelation inequality task."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50
DEADLINE = 120.0  # seconds

def compute_value(h):
    """Compute the value metric for given heights h."""
    h = np.asarray(h, dtype=float)
    if h.shape != (N,):
        return 0.0
    if np.any(~np.isfinite(h)) or np.any(h < -1e-12):
        return 0.0
    h = np.clip(h, 0.0, None)
    
    g = np.convolve(h, h, mode='full')
    m = len(g)
    y = np.concatenate(([0.0], g, [0.0]))
    
    w = 1.0 / (N + 1)  # spacing: interval length 1.0 / (N+1) subintervals
    # Actually, the points are np.linspace(-0.5, 0.5, m+2), so spacing = 1.0/(m+1)
    # m = 2*N - 1, so m+1 = 2*N, spacing = 1/(2*N)
    w = 1.0 / (m + 1)
    
    # l2 = sum over i in 0..m of (w/3) * (y[i]^2 + y[i]*y[i+1] + y[i+1]^2)
    l2 = (w / 3.0) * np.sum(y[:m+1]**2 + y[:m+1] * y[1:m+2] + y[1:m+2]**2)
    
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    
    if norm_1 * norm_inf < 1e-30:
        return 0.0
    
    return l2 / (norm_1 * norm_inf)

def neg_value(params):
    """Negative value for minimization. params are DCT coefficients."""
    # Reconstruct h from DCT coefficients with softplus to ensure non-negativity
    coeffs = params
    # Use DCT type II inverse: h = dct(coeffs, type=2, norm='ortho')
    h = np.fft.irfft(np.abs(np.fft.rfft(coeffs)) ** 2)  # This doesn't work well
    
    # Better: use scipy's dct
    from scipy.fftpack import dct, idct
    h = idct(coeffs, type=2, norm='ortho')
    h = np.maximum(h, 0.0)
    
    if np.any(~np.isfinite(h)):
        return 1e10
    
    val = compute_value(h)
    return -val

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE - 5.0  # 5 second safety margin
    
    # Initial: uniform
    best_h = np.ones(N)
    best_val = compute_value(best_h)
    
    # Use DCT parametrization for optimization
    # h = idct(coeffs, type=2, norm='ortho'), then clip to non-negative
    # Start with coefficients that give uniform
    from scipy.fftpack import dct, idct
    init_coeffs = dct(best_h, type=2, norm='ortho')
    
    # Try several optimization strategies
    strategies = [
        {'method': 'L-BFGS-B', 'options': {'maxiter': 200, 'maxfun': 500}},
        {'method': 'Nelder-Mead', 'options': {'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6}},
        {'method': 'Powell', 'options': {'maxiter': 100, 'maxfev': 1000}},
    ]
    
    for strat in strategies:
        if time.time() > deadline:
            break
        
        try:
            res = minimize(neg_value, init_coeffs, method=strat['method'], 
                          options=strat['options'], timeout=30.0)
            
            if res.success or res.fun < -best_val:
                coeffs = res.x
                h = idct(coeffs, type=2, norm='ortho')
                h = np.maximum(h, 0.0)
                val = compute_value(h)
                if val > best_val:
                    best_val = val
                    best_h = h
                    # Update init for next strategy
                    init_coeffs = dct(best_h, type=2, norm='ortho')
        except Exception:
            continue
    
    # Local refinement with direct optimization
    if time.time() < deadline - 10.0:
        def neg_value_direct(h):
            h = np.maximum(h, 0.0)
            if np.any(~np.isfinite(h)):
                return 1e10
            return -compute_value(h)
        
        for _ in range(3):
            if time.time() > deadline:
                break
            try:
                res = minimize(neg_value_direct, best_h, method='L-BFGS-B',
                              bounds=[(0, None)] * N,
                              options={'maxiter': 100, 'maxfun': 200},
                              timeout=20.0)
                if res.fun < -best_val:
                    h = np.maximum(res.x, 0.0)
                    val = compute_value(h)
                    if val > best_val:
                        best_val = val
                        best_h = h
            except Exception:
                continue
    
    # Ensure non-negative and finite
    best_h = np.clip(best_h, 0.0, None)
    best_h = np.where(np.isfinite(best_h), best_h, 0.0)
    
    return list(best_h)
