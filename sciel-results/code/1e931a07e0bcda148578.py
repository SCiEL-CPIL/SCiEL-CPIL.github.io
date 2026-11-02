"""Program using model-and-optimize approach for first autocorrelation inequality."""
import time
import numpy as np
from scipy import optimize

N = 600

def _fft_convolve(h):
    """Fast linear convolution via FFT."""
    n = len(h)
    size = 2 * n - 1
    # Next power of 2 for FFT efficiency
    fft_size = 1
    while fft_size < size:
        fft_size *= 2
    H = np.fft.rfft(h, fft_size)
    conv = np.fft.irfft(H * H, fft_size)[:size]
    return conv

def compute_score(h):
    """
    Compute the normalized maximum of the autoconvolution.
    h: array of nonnegative values.
    Returns max_k (h*h)[k] / (sum h)^2
    """
    s = np.sum(h)
    if s == 0:
        return 0.0
    conv = _fft_convolve(h)
    max_conv = np.max(conv)
    return max_conv / (s * s)

def compute_score_smooth(h, beta=100.0):
    """
    Smooth surrogate for compute_score using log-sum-exp.
    This is differentiable and suitable for gradient-based optimization.
    """
    s = np.sum(h)
    if s == 0:
        return 0.0
    conv = _fft_convolve(h)
    # Log-sum-exp for smooth max
    max_val = np.max(conv)
    if max_val <= 0:
        return 0.0
    # Smooth max: (1/beta) * log(sum(exp(beta * conv / max_val))) * max_val / s^2
    # Simpler: use softplus-like approximation
    shifted = conv - max_val
    log_sum_exp = max_val + np.log(np.sum(np.exp(shifted * beta))) / beta
    return log_sum_exp / (s * s)

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # We want to minimize max_k conv[k] / (sum h)^2 subject to h[i] >= 0.
    # This is scale invariant. We can fix sum(h) = 1.
    # The problem is equivalent to finding a probability mass function p on {0,...,N-1}
    # that minimizes the peak of its linear convolution (p * p).
    # Note: linear convolution of length N vectors has length 2N-1.
    
    # Coordinates x_i:
    # The intervals are equal width on [-1/4, 1/4].
    # Width w = 0.5 / 600 = 1/1200.
    # Center of interval i: x_i = -0.25 + (i + 0.5) * w.
    
    w = 0.5 / N
    x = np.linspace(-0.25 + w/2, 0.25 - w/2, N)
    
    # Family 1: h(x) = exp(-(|x| / sigma)^p)
    def param_h1(sigma, p):
        if sigma <= 0 or p <= 0:
            return np.ones(N) / N
        h = np.exp(-np.abs(x / sigma) ** p)
        s = np.sum(h)
        if s > 0:
            h = h / s
        return h
    
    # Family 2: h(x) = (1 - |x|/L)^a for |x| < L, 0 otherwise
    # L controls support width (fraction of half-interval)
    def param_h2(a, L_frac):
        if a <= 0 or L_frac <= 0:
            return np.ones(N) / N
        L = L_frac * 0.25  # L_frac in (0, 1], L in (0, 0.25]
        abs_x = np.abs(x)
        h = np.zeros(N)
        mask = abs_x < L
        h[mask] = (1 - abs_x[mask] / L) ** a
        s = np.sum(h)
        if s > 0:
            h = h / s
        return h
    
    # Family 3: h(x) = (1 - (|x|/L)^a)^b for |x| < L, 0 otherwise
    def param_h3(a, b, L_frac):
        if a <= 0 or b <= 0 or L_frac <= 0:
            return np.ones(N) / N
        L = L_frac * 0.25
        abs_x = np.abs(x)
        h = np.zeros(N)
        mask = abs_x < L
        t = (abs_x[mask] / L) ** a
        h[mask] = (1 - t) ** b
        s = np.sum(h)
        if s > 0:
            h = h / s
        return h
    
    # Smooth surrogate objective using log-sum-exp
    def smooth_objective(h, beta=50.0):
        s = np.sum(h)
        if s == 0:
            return 0.0
        conv = _fft_convolve(h)
        max_val = np.max(conv)
        if max_val <= 0:
            return 0.0
        shifted = conv - max_val
        # Use a stable log-sum-exp
        log_sum = max_val + np.log(np.sum(np.exp(shifted * beta))) / beta
        return log_sum / (s * s)
    
    # Initialize best
    best_h = np.ones(N) / N
    best_score = compute_score(best_h)
    
    # Grid search over family 1
    sigmas = [0.03, 0.05, 0.08, 0.1, 0.12, 0.15, 0.2, 0.25]
    ps = [0.5, 1, 1.5, 2, 3, 4, 6]
    
    for s0 in sigmas:
        for p0 in ps:
            if time.time() > deadline:
                break
            h0 = param_h1(s0, p0)
            score0 = compute_score(h0)
            if score0 < best_score:
                best_score = score0
                best_h = h0.copy()
    
    # Grid search over family 2
    a_vals2 = [0.5, 1, 1.5, 2, 3, 5]
    L_fracs = [0.3, 0.5, 0.7, 0.9, 1.0]
    
    for a0 in a_vals2:
        for L0 in L_fracs:
            if time.time() > deadline:
                break
            h0 = param_h2(a0, L0)
            score0 = compute_score(h0)
            if score0 < best_score:
                best_score = score0
                best_h = h0.copy()
    
    # Grid search over family 3
    a_vals3 = [0.5, 1, 2, 3, 4]
    b_vals3 = [0.5, 1, 2, 3, 4, 6]
    L_fracs3 = [0.5, 0.7, 0.9, 1.0]
    
    for a0 in a_vals3:
        for b0 in b_vals3:
            for L0 in L_fracs3:
                if time.time() > deadline:
                    break
                h0 = param_h3(a0, b0, L0)
                score0 = compute_score(h0)
                if score0 < best_score:
                    best_score = score0
                    best_h = h0.copy()
    
    # Local optimization using smooth surrogate
    # Family 1 optimization
    def obj1_smooth(params):
        sigma, p = params
        sigma = np.clip(sigma, 0.01, 0.5)
        p = np.clip(p, 0.3, 10.0)
        h = param_h1(sigma, p)
        return smooth_objective(h, beta=50.0)
    
    # Family 2 optimization
    def obj2_smooth(params):
        a, L_frac = params
        a = np.clip(a, 0.1, 10.0)
        L_frac = np.clip(L_frac, 0.1, 1.0)
        h = param_h2(a, L_frac)
        return smooth_objective(h, beta=50.0)
    
    # Family 3 optimization
    def obj3_smooth(params):
        a, b, L_frac = params
        a = np.clip(a, 0.1, 10.0)
        b = np.clip(b, 0.1, 10.0)
        L_frac = np.clip(L_frac, 0.1, 1.0)
        h = param_h3(a, b, L_frac)
        return smooth_objective(h, beta=50.0)
    
    # Optimize family 1 from best grid points
    max_opts = 15
    opt_count = 0
    
    for s0 in sigmas:
        for p0 in ps:
            if time.time() > deadline or opt_count >= max_opts:
                break
            opt_count += 1
            bounds = [(0.01, 0.5), (0.3, 10.0)]
            try:
                res = optimize.minimize(
                    obj1_smooth, 
                    [s0, p0], 
                    method='L-BFGS-B', 
                    bounds=bounds,
                    options={'maxiter': 30, 'ftol': 1e-9}
                )
                if res.fun < best_score:
                    h_candidate = param_h1(res.x[0], res.x[1])
                    actual_score = compute_score(h_candidate)
                    if actual_score < best_score:
                        best_score = actual_score
                        best_h = h_candidate
            except Exception:
                pass
    
    # Optimize family 2
    for a0 in a_vals2:
        for L0 in L_fracs:
            if time.time() > deadline or opt_count >= max_opts + 10:
                break
            opt_count += 1
            bounds = [(0.1, 10.0), (0.1, 1.0)]
            try:
                res = optimize.minimize(
                    obj2_smooth, 
                    [a0, L0], 
                    method='L-BFGS-B', 
                    bounds=bounds,
                    options={'maxiter': 30, 'ftol': 1e-9}
                )
                if res.fun < best_score:
                    h_candidate = param_h2(res.x[0], res.x[1])
                    actual_score = compute_score(h_candidate)
                    if actual_score < best_score:
                        best_score = actual_score
                        best_h = h_candidate
            except Exception:
                pass
    
    # Optimize family 3
    for a0 in a_vals3:
        for b0 in b_vals3:
            for L0 in L_fracs3:
                if time.time() > deadline or opt_count >= max_opts + 20:
                    break
                opt_count += 1
                bounds = [(0.1, 10.0), (0.1, 10.0), (0.1, 1.0)]
                try:
                    res = optimize.minimize(
                        obj3_smooth, 
                        [a0, b0, L0], 
                        method='L-BFGS-B', 
                        bounds=bounds,
                        options={'maxiter': 20, 'ftol': 1e-9}
                    )
                    if res.fun < best_score:
                        h_candidate = param_h3(res.x[0], res.x[1], res.x[2])
                        actual_score = compute_score(h_candidate)
                        if actual_score < best_score:
                            best_score = actual_score
                            best_h = h_candidate
                except Exception:
                    pass
    
    # Final polish: if we have time, do a high-precision optimization from best
    if time.time() < deadline - 10:
        # Try to refine using the exact objective with tighter tolerance
        # Re-optimize family 1 and 2 from current best with exact score
        def obj1_exact(params):
            sigma, p = params
            sigma = np.clip(sigma, 0.01, 0.5)
            p = np.clip(p, 0.3, 10.0)
            h = param_h1(sigma, p)
            return compute_score(h)
        
        def obj2_exact(params):
            a, L_frac = params
            a = np.clip(a, 0.1, 10.0)
            L_frac = np.clip(L_frac, 0.1, 1.0)
            h = param_h2(a, L_frac)
            return compute_score(h)
        
        # Try a few refinements
        for s0 in [best_h if False else 0.1, 0.15, 0.2]:
            for p0 in [1, 2, 3]:
                if time.time() > deadline - 5:
                    break
                bounds = [(0.01, 0.5), (0.3, 10.0)]
                try:
                    res = optimize.minimize(
                        obj1_exact, 
                        [s0, p0], 
                        method='L-BFGS-B', 
                        bounds=bounds,
                        options={'maxiter': 20, 'ftol': 1e-10}
                    )
                    if res.fun < best_score:
                        h_candidate = param_h1(res.x[0], res.x[1])
                        actual_score = compute_score(h_candidate)
                        if actual_score < best_score:
                            best_score = actual_score
                            best_h = h_candidate
                except Exception:
                    pass
    
    # Final check: ensure best_h is valid (non-negative)
    best_h = np.maximum(0, best_h)
    
    return best_h.tolist()
# EVOLVE-BLOCK-END
