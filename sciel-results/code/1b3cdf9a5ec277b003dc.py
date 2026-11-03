"""Optimize step function to minimize normalized max autoconvolution."""
import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

N = 600
DELTA = 1.0 / (4 * N)  # interval width: [-1/4, 1/4] has length 1/2, so each interval is 1/(2*600) = 1/1200
# Actually [-1/4, 1/4] has length 1/2, so 600 intervals each of width 1/1200

def _compute_normalized_max(h):
    """Compute normalized maximum of autoconvolution for step heights h."""
    # Autoconvolution via FFT
    # The autoconvolution of a step function: 
    # (f*f)(k*delta) = delta * sum_i h_i * h_{i+k} for appropriate indexing
    # More precisely, for discrete autoconvolution:
    n = len(h)
    # Use FFT to compute linear autoconvolution
    size = 2 * n
    H = np.fft.rfft(h, size)
    conv = np.fft.irfft(H * H, size)[:n]  # linear autoconvolution, first n values
    # The autoconvolution at lag 0 is sum(h^2), at lag k is sum h_i * h_{i+k}
    # Normalize: max of conv / (sum(h))^2 * (1/delta) ... 
    # Actually the continuous autoconvolution max divided by (integral f)^2
    # integral f = delta * sum(h)
    # autoconvolution(t) = integral f(s)f(t-s)ds
    # At the discrete level: conv[k] * delta approximates the autoconvolution at t = k*delta
    # So normalized max = max(conv) * delta / (delta * sum(h))^2 = max(conv) / (delta * (sum(h))^2)
    # Wait, let me be more careful.
    # f(s) = h_i for s in [(i-0.5)*delta, (i+0.5)*delta) approximately
    # (f*f)(t) = integral f(s) f(t-s) ds
    # For t = k*delta, this is approximately delta * sum_i h_i * h_{i-k} (with appropriate bounds)
    # So (f*f)(k*delta) ≈ delta * conv[k] where conv[k] = sum_i h_i * h_{i-k}
    # The integral of f is delta * sum(h)
    # Normalized max = max_k (f*f)(k*delta) / (integral f)^2 = delta * max(conv) / (delta^2 * (sum(h))^2)
    # = max(conv) / (delta * (sum(h))^2)
    s = np.sum(h)
    if s == 0:
        return 0.0
    max_conv = np.max(conv)
    return max_conv / (DELTA * s * s)

def _param_to_heights(params):
    """Convert parameters to step heights."""
    # params: [amplitude, center, width, skew]
    # Use a smooth bump: A * exp(-((x-center)^2)/(2*width^2)) * (1 + skew*(x-center)/width)
    # clipped to nonnegative
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25 / N  # centers of intervals
    A, center, width, skew = params
    if width <= 0:
        width = 1e-6
    heights = A * np.exp(-0.5 * ((x - center) / width) ** 2) * (1.0 + skew * (x - center) / width)
    heights = np.maximum(heights, 0.0)
    return heights

def _score_from_params(params):
    h = _param_to_heights(params)
    return _compute_normalized_max(h)

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    best_score = 1.0  # uniform gives 1.0
    best_h = np.ones(N)
    
    # Try a few initial parametric profiles
    # 1. Gaussian centered
    # 2. Cosine bump
    # 3. Triangular
    
    def _gaussian_score(params):
        A, center, width = params
        x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25 / N
        h = A * np.exp(-0.5 * ((x - center) / max(width, 1e-6)) ** 2)
        h = np.maximum(h, 0.0)
        return _compute_normalized_max(h)
    
    def _cosine_score(params):
        A, center, width = params
        x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25 / N
        z = np.abs(x - center) / max(width, 1e-6)
        h = A * np.where(z < 1, 0.5 * (1 + np.cos(np.pi * z)), 0.0)
        h = np.maximum(h, 0.0)
        return _compute_normalized_max(h)
    
    # Quick grid search over Gaussian parameters
    for A in [0.5, 1.0, 2.0, 5.0, 10.0]:
        for center in [-0.1, 0.0, 0.1]:
            for width in [0.02, 0.05, 0.1, 0.15, 0.2]:
                if time.time() > deadline:
                    break
                score = _gaussian_score([A, center, width])
                if score < best_score:
                    best_score = score
                    best_h = A * np.exp(-0.5 * ((np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25/N - center) / width) ** 2)
                    best_h = np.maximum(best_h, 0.0)
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    # Quick grid search over Cosine parameters
    for A in [0.5, 1.0, 2.0, 5.0, 10.0]:
        for center in [-0.1, 0.0, 0.1]:
            for width in [0.02, 0.05, 0.1, 0.15, 0.2]:
                if time.time() > deadline:
                    break
                score = _cosine_score([A, center, width])
                if score < best_score:
                    best_score = score
                    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25 / N
                    z = np.abs(x - center) / width
                    best_h = A * np.where(z < 1, 0.5 * (1 + np.cos(np.pi * z)), 0.0)
                    best_h = np.maximum(best_h, 0.0)
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    # Local refinement with Nelder-Mead on the best parametric form
    if time.time() < deadline - 20:
        # Refine Gaussian
        x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25 / N
        # Try to refine around the best found
        # Use a few starting points
        for A0 in [1.0, 5.0]:
            for c0 in [0.0]:
                for w0 in [0.1, 0.15]:
                    if time.time() > deadline - 10:
                        break
                    try:
                        res = minimize(_gaussian_score, [A0, c0, w0], method='Nelder-Mead',
                                      options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10})
                        if res.fun < best_score:
                            best_score = res.fun
                            h = A0 * np.exp(-0.5 * ((x - c0) / w0) ** 2)
                            # Recompute with optimal params
                            A_opt, c_opt, w_opt = res.x
                            h = A_opt * np.exp(-0.5 * ((x - c_opt) / max(w_opt, 1e-6)) ** 2)
                            h = np.maximum(h, 0.0)
                            best_h = h
                    except:
                        pass
                if time.time() > deadline - 10:
                    break
            if time.time() > deadline - 10:
                break
    
    # Refine Cosine
    if time.time() < deadline - 10:
        for A0 in [1.0, 5.0]:
            for c0 in [0.0]:
                for w0 in [0.1, 0.15]:
                    if time.time() > deadline - 5:
                        break
                    try:
                        res = minimize(_cosine_score, [A0, c0, w0], method='Nelder-Mead',
                                      options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10})
                        if res.fun < best_score:
                            best_score = res.fun
                            x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25 / N
                            A_opt, c_opt, w_opt = res.x
                            z = np.abs(x - c_opt) / max(w_opt, 1e-6)
                            h = A_opt * np.where(z < 1, 0.5 * (1 + np.cos(np.pi * z)), 0.0)
                            h = np.maximum(h, 0.0)
                            best_h = h
                    except:
                        pass
                if time.time() > deadline - 5:
                    break
            if time.time() > deadline - 5:
                break
    
    # Final check: ensure nonnegative and return
    best_h = np.maximum(best_h, 0.0)
    
    # If best is still 1.0, maybe try a direct small perturbation approach
    if best_score >= 1.0 and time.time() < deadline - 5:
        # Try a simple interior point: make the function slightly peaked
        x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25 / N
        for peak_factor in [1.1, 1.2, 1.5, 2.0]:
            h = np.ones(N) * (1.0 / peak_factor)  # scale down
            h = h * (1.0 + (peak_factor - 1.0) * np.exp(-0.5 * (x / 0.1) ** 2))
            h = np.maximum(h, 0.0)
            score = _compute_normalized_max(h)
            if score < best_score:
                best_score = score
                best_h = h
    
    return list(best_h)
