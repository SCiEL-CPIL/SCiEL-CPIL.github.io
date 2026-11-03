"""
First autocorrelation inequality: minimize normalized max of autoconvolution
for a nonnegative step function with 600 equal intervals on [-1/4, 1/4].

Strategy: Use a smooth parametric family based on cosine bumps, then refine
with an FFT-accelerated sensitivity-based local optimization that targets peak reduction.

Key improvements:
1. FFT-based convolution for speed (enables many more refinement iterations)
2. Vectorized sensitivity computation via FFT
3. Adaptive step size with better convergence
4. Multiple restarts with different parametric initializations
"""

import numpy as np
from scipy.optimize import minimize
import time

N = 600
TIME_LIMIT = 140  # safety margin before 150s kill

def fft_convolve(f):
    """Compute autoconvolution using FFT for speed."""
    n = 2 * N
    F = np.fft.rfft(f, n=n)
    conv = np.fft.irfft(F * F, n=n)
    return conv[:2*N-1]

def compute_score(f):
    """Compute the normalized maximum of the autoconvolution."""
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return np.inf
    f = np.maximum(f, 0.0)
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    conv = fft_convolve(f)
    return np.max(conv) / (s * s)

def smooth_max(x, beta):
    """Smooth approximation to max using log-sum-exp."""
    x = np.asarray(x, dtype=np.float64)
    x_max = np.max(x)
    if x_max < 1e-15:
        return 0.0
    x_scaled = x / x_max
    shifted = x_scaled - np.max(x_scaled)
    log_sum = np.log(np.sum(np.exp(beta * shifted))) / beta
    return x_max * (1.0 + log_sum)

def cosine_basis(x, centers, widths, heights):
    """Generate a smooth nonnegative function from cosine bump components."""
    result = np.zeros_like(x)
    for c, w, h in zip(centers, widths, heights):
        dist = np.abs(x - c)
        mask = dist <= w / 2.0
        if np.any(mask):
            result[mask] += h * np.cos(np.pi * (x[mask] - c) / w) ** 2
    return result

def parametric_to_step(params, n_components):
    """Convert parameters to a step function of length N."""
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25 / N
    centers = params[:n_components]
    widths = params[n_components:2*n_components]
    heights = params[2*n_components:3*n_components]
    
    widths = np.abs(widths) + 0.01
    widths = np.minimum(widths, 0.5)
    heights = np.abs(heights)
    
    f = cosine_basis(x, centers, widths, heights)
    f = np.maximum(f, 0.0)
    return f

def objective(params, n_components, beta=10.0):
    f = parametric_to_step(params, n_components)
    conv = fft_convolve(f)
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    return smooth_max(conv, beta) / (s * s)

def local_refinement(h, start_time, max_time):
    """Local refinement using sensitivity-based peak reduction with FFT.
    
    At each step, we identify lags where the autoconvolution is near its max,
    compute how much each bin contributes to those peaks, and reduce bins
    that contribute more to peaks than to off-peak regions.
    Uses FFT for fast convolution and vectorized sensitivity computation.
    """
    best_h = h.copy()
    best_val = compute_score(h)
    current_h = h.copy()
    alpha = 0.05
    threshold_frac = 0.95
    
    # Precompute FFT size
    n_fft = 2 * N
    
    while time.time() - start_time < max_time:
        s = np.sum(current_h)
        if s < 1e-12:
            break
        # Normalize to sum = N for stable computation
        current_h = current_h * (N / s)
        
        conv = fft_convolve(current_h)
        max_c = np.max(conv)
        current_val = max_c / (N * N)
        
        if current_val < best_val:
            best_val = current_val
            best_h = current_h.copy()
        
        # Find lags where conv is close to max
        threshold = max_c * threshold_frac
        peak_indices = np.where(conv >= threshold)[0]
        
        # Compute sensitivity using FFT:
        # conv[m] = sum_j f[j] * f[m-j]
        # d(conv[m])/d(f[j]) = f[m-j]
        # For peak lags M = {m1, m2, ...}, sensitivity[j] = sum_{m in M} f[m-j]
        # This is a cross-correlation of f with the indicator of peak lags
        peak_indicator = np.zeros(2*N-1)
        peak_indicator[peak_indices] = 1.0
        
        # sensitivity[j] = sum_m peak_indicator[m] * f[m-j]
        # This is the cross-correlation of f with peak_indicator
        # Using FFT: ifft(fft(f) * conj(fft(peak_indicator)))
        F = np.fft.rfft(current_h, n=n_fft)
        P = np.fft.rfft(peak_indicator, n=n_fft)
        # Cross-correlation: ifft(F * conj(P))
        corr = np.fft.irfft(F * np.conj(P), n=n_fft)
        # corr[k] = sum_j f[j] * peak_indicator[j+k] (circular)
        # We want sensitivity[j] = sum_m peak_indicator[m] * f[m-j]
        # = sum_m peak_indicator[m] * f[m-j]
        # Let k = m-j, so m = j+k: sum_k peak_indicator[j+k] * f[k]
        # This is the cross-correlation at lag j
        # corr[j] in the above gives sum_k f[k] * peak_indicator[j+k]
        # So sensitivity = corr (appropriately indexed)
        sensitivity = corr[:N]
        
        # Normalize sensitivity
        sens_mean = np.mean(sensitivity)
        sens_std = np.std(sensitivity) + 1e-12
        sens_norm = (sensitivity - sens_mean) / sens_std
        
        # Reduce bins with high sensitivity (they contribute to peaks)
        h_new = current_h * (1.0 - alpha * sens_norm)
        h_new = np.maximum(h_new, 0.0)
        
        s_new = np.sum(h_new)
        if s_new < 1e-12:
            break
        h_new = h_new * (N / s_new)
        
        conv_new = fft_convolve(h_new)
        val_new = np.max(conv_new) / (N * N)
        
        if val_new < current_val:
            current_h = h_new
            alpha = min(alpha * 1.05, 0.5)
        else:
            alpha *= 0.85
            if alpha < 1e-8:
                break
    
    return best_h, best_val

def spectral_polish(h, start_time, max_time):
    """High-precision polish pass using subgradient method with exact max.
    
    Uses a subgradient approach: at each step, compute the exact maximum
    of the autoconvolution, identify all lags achieving near-max, and
    adjust h to reduce those peaks.
    """
    best_h = h.copy()
    best_val = compute_score(h)
    current_h = h.copy()
    lr = 0.01
    
    while time.time() - start_time < max_time:
        s = np.sum(current_h)
        if s < 1e-12:
            break
        current_h = current_h * (N / s)
        
        conv = fft_convolve(current_h)
        max_c = np.max(conv)
        current_val = max_c / (N * N)
        
        if current_val < best_val:
            best_val = current_val
            best_h = current_h.copy()
        
        # Identify peak lags (within 1% of max)
        threshold = max_c * 0.99
        peak_mask = conv >= threshold
        
        # Subgradient: for the max of conv, the subgradient w.r.t. h[j] is
        # sum_{m: conv[m] in max} h[m-j]
        # We want to decrease h in directions that increase peak conv values
        peak_indicator = peak_mask.astype(np.float64)
        
        F = np.fft.rfft(current_h, n=2*N)
        P = np.fft.rfft(peak_indicator, n=2*N)
        corr = np.fft.irfft(F * np.conj(P), n=2*N)
        sensitivity = corr[:N]
        
        # Normalize and apply update
        sens_norm = sensitivity / (np.max(np.abs(sensitivity)) + 1e-12)
        h_new = current_h - lr * sens_norm
        h_new = np.maximum(h_new, 0.0)
        
        s_new = np.sum(h_new)
        if s_new < 1e-12:
            break
        h_new = h_new * (N / s_new)
        
        conv_new = fft_convolve(h_new)
        val_new = np.max(conv_new) / (N * N)
        
        if val_new < current_val:
            current_h = h_new
            lr = min(lr * 1.02, 0.1)
        else:
            lr *= 0.9
            if lr < 1e-10:
                break
    
    return best_h, best_val

def construct():
    start_time = time.time()
    
    # Initial best: constant function
    best_f = np.ones(N)
    best_score = compute_score(best_f)
    
    # Try different numbers of cosine bump components
    for n_c in [2, 3, 4, 5]:
        if time.time() - start_time > TIME_LIMIT - 40:
            break
        
        rng = np.random.RandomState(42 + n_c)
        best_params = None
        best_obj = np.inf
        
        # Annealed beta: start lower, increase for sharper max approximation
        betas = [5.0, 10.0, 20.0]
        
        for beta in betas:
            if time.time() - start_time > TIME_LIMIT - 20:
                break
            
            for restart in range(4):
                if time.time() - start_time > TIME_LIMIT - 15:
                    break
                
                centers = rng.uniform(-0.2, 0.2, size=n_c)
                widths = rng.uniform(0.05, 0.2, size=n_c)
                heights = rng.uniform(0.5, 2.0, size=n_c)
                
                x0 = np.concatenate([centers, widths, heights])
                
                try:
                    result = minimize(
                        objective, x0, args=(n_c, beta),
                        method='Nelder-Mead',
                        options={'maxiter': 1500, 'xatol': 1e-6, 'fatol': 1e-8}
                    )
                    if result.fun < best_obj:
                        best_obj = result.fun
                        best_params = result.x
                except Exception:
                    pass
        
        if best_params is not None:
            f = parametric_to_step(best_params, n_c)
            actual_score = compute_score(f)
            if actual_score < best_score:
                best_f = f
                best_score = actual_score
    
    # Local refinement: sensitivity-based peak reduction
    if time.time() - start_time < TIME_LIMIT - 20:
        refined_f, refined_score = local_refinement(
            best_f, start_time, TIME_LIMIT - 10
        )
        if refined_score < best_score:
            best_f = refined_f
            best_score = refined_score
    
    # Spectral polish for high-precision final improvement
    if time.time() - start_time < TIME_LIMIT - 5:
        polished_f, polished_score = spectral_polish(
            best_f, start_time, TIME_LIMIT - 2
        )
        if polished_score < best_score:
            best_f = polished_f
            best_score = polished_score
    
    # Ensure nonnegative
    best_f = np.maximum(best_f, 0.0)
    return [float(x) for x in best_f]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
