import numpy as np
import time
from scipy.optimize import minimize

N = 600
DELTA = 1.0 / N  # interval width

def compute_autoconv_max(h):
    """Compute max of autoconvolution of step function h (heights on N equal intervals)."""
    # Autoconvolution via FFT for efficiency
    # f is defined on [-1/4, 1/4] with N intervals
    # f*f is defined on [-1/2, 1/2]
    # Sample f*f at the N+1 interval boundaries? Or at more points?
    # For step functions, f*f is piecewise linear, max occurs at interval boundaries or where slope changes
    # Simplest: compute on a fine grid using convolution
    
    # Use np.convolve for the discrete autoconvolution
    # The continuous autoconvolution sampled at grid points:
    # (f*f)(k*DELTA) ≈ DELTA * sum_i h[i] * h[i-k] for appropriate k
    # Actually for step functions: (f*f)(t) for t in [(k-1)*Δ, k*Δ] is linear
    # Max of piecewise linear function on this domain occurs at breakpoints
    
    # Let's compute on a fine grid
    # f*f has support on [-1/2, 1/2], which is 2N intervals of width Δ
    # Compute at all 2N+1 interval boundaries
    h_float = np.asarray(h, dtype=np.float64)
    # Discrete convolution: c[k] = sum_i h[i]*h[i-k]
    # (f*f)(t) for t = k*Δ (k from -N+1 to N-1) 
    # Actually let's just do full convolution
    conv = np.convolve(h_float, h_float, mode='full')
    # conv has length 2N-1, conv[k] corresponds to lag k-(N-1)
    # The continuous autoconvolution max: for step functions, the max of the piecewise linear
    # function occurs at the midpoints of the linear pieces or at breakpoints
    # Breakpoints are at multiples of Δ. Between breakpoints it's linear.
    # So max is at a breakpoint or... for a convex piece (like center) max is at center of interval
    # Let's evaluate at both breakpoints and midpoints to be safe
    
    # conv[j] for j=0..2N-2 gives sum_{i} h[i]*h[i-(j-(N-1))]
    # This is the value of the discrete convolution at lag = j-(N-1)
    # The continuous autoconvolution at t = lag*DELTA is DELTA * conv_value? 
    # Let me think: f(x) = h[i] for x in [i*Δ - 1/4, (i+1)*Δ - 1/4)
    # (f*f)(t) = ∫ f(t-s)f(s) ds
    # For step functions, this is piecewise linear. The value at t = k*Δ can be computed.
    
    # Simpler: just find max of conv (which gives the peak of the discrete approximation)
    # and scale appropriately. The key is the RATIO, so normalization cancels.
    
    max_val = np.max(conv)
    l1_norm_sq = (np.sum(h_float) * DELTA) ** 2
    if l1_norm_sq < 1e-15:
        return 1e15
    return max_val * DELTA / l1_norm_sq  # normalize: max(f*f) / (∫f)^2

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin
    
    # Start with uniform
    h = np.ones(N)
    
    # Try to improve via coordinate descent / local optimization
    # Objective: minimize max(f*f) / (∫f)^2
    # We can fix the L1 norm to 1 by normalizing, so minimize max(f*f) subject to sum(h)*Δ = 1, h >= 0
    
    def objective(h_vec):
        h_nonneg = np.maximum(h_vec, 0)
        s = np.sum(h_nonneg) * DELTA
        if s < 1e-15:
            return 1e15
        h_norm = h_nonneg / s  # normalize to L1 = 1
        conv = np.convolve(h_norm, h_norm, mode='full')
        return np.max(conv) * DELTA  # max of autoconvolution
    
    best_h = h.copy()
    best_obj = objective(best_h)
    
    # Coordinate descent: adjust one coordinate at a time
    # For each coordinate i, try to find optimal h[i] given others fixed
    # With L1 normalization, this is tricky. Let's use a different approach.
    
    # Alternative: parameterize as h_i = exp(x_i) / sum(exp(x_j)) to ensure nonnegativity
    # and fixed L1. Then minimize max(f*f).
    
    # Actually, let's try a simpler heuristic: 
    # The optimal shape for minimizing max(f*f) with fixed support and L1 norm
    # might be close to uniform, but slightly modified.
    
    # Let's try scipy minimize with the log-parameterization
    from scipy.optimize import minimize as sp_minimize
    
    def obj_log(x):
        h_exp = np.exp(x - np.max(x))  # for numerical stability
        h_norm = h_exp / np.sum(h_exp)  # L1 normalized
        conv = np.convolve(h_norm, h_norm, mode='full')
        return np.max(conv) * DELTA
    
    # Start from uniform: all x_i = 0
    x0 = np.zeros(N)
    
    # Try a few optimization methods
    best_val = best_obj
    best_h = h.copy()
    
    elapsed = time.time() - start_time
    if elapsed < deadline - 30:
        try:
            res = sp_minimize(obj_log, x0, method='Nelder-Mead', 
                            options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10, 'maxfev': 10000})
            if res.fun < best_val:
                best_val = res.fun
                x_best = res.x
                h_exp = np.exp(x_best - np.max(x_best))
                best_h = h_exp / np.sum(h_exp)
        except Exception:
            pass
    
    elapsed = time.time() - start_time
    if elapsed < deadline - 20:
        # Try coordinate descent on the original h with L1 constraint
        # Normalize h to have L1 = 1
        s = np.sum(best_h) * DELTA
        if s > 1e-15:
            best_h = best_h / s
        
        for iteration in range(200):
            if time.time() - start_time > deadline - 10:
                break
            improved = False
            for i in range(N):
                # Try changing h[i]
                old_val = best_h[i]
                # The objective is max of a piecewise linear function in h[i]
                # Try a few candidate values
                for candidate in [old_val * 0.5, old_val * 0.8, old_val * 1.2, old_val * 1.5, 0.0, old_val * 2.0]:
                    candidate = max(candidate, 0.0)
                    h_test = best_h.copy()
                    h_test[i] = candidate
                    # Renormalize to L1 = 1
                    s_test = np.sum(h_test) * DELTA
                    if s_test < 1e-15:
                        continue
                    h_test = h_test / s_test
                    conv_test = np.convolve(h_test, h_test, mode='full')
                    obj_test = np.max(conv_test) * DELTA
                    if obj_test < best_val - 1e-12:
                        best_val = obj_test
                        best_h = h_test
                        improved = True
                        break
    
    # Final: return as list of floats
    # Ensure nonnegative
    best_h = np.maximum(best_h, 0.0)
    
    # Check deadline
    if time.time() - start_time > deadline:
        return [1.0] * N
    
    return [float(x) for x in best_h]

# EVOLVE-BLOCK-END
