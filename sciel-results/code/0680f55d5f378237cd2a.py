"""
First autocorrelation inequality: minimize normalized max of autoconvolution
for a nonnegative step function with 600 equal intervals on [-1/4, 1/4].

Strategy: Use a smooth parametric family based on cosine bumps to get a good
initial shape, then refine with a fast FFT-accelerated sensitivity-based peak
reduction. The key insight from the inspirations is to use the FFT trick to
compute sensitivity in O(N log N) instead of O(N * n_peaks), which allows
many more refinement steps within the time budget. We also add a spectral
flattening pass to further reduce peaks.
"""

import numpy as np
from scipy.optimize import minimize
import time

N = 600
TIME_LIMIT = 140  # safety margin before 150s kill

def fft_convolve(f, g):
    """FFT-based convolution for speed."""
    n = len(f) + len(g) - 1
    size = 1
    while size < n:
        size *= 2
    F = np.fft.rfft(f, size)
    G = np.fft.rfft(g, size)
    return np.fft.irfft(F * G, size)[:n]

def compute_score(f):
    """Compute the normalized maximum of the autoconvolution."""
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return np.inf
    f = np.maximum(f, 0.0)
    conv = fft_convolve(f, f)
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
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
    conv = fft_convolve(f, f)
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    return smooth_max(conv, beta) / (s * s)

def local_refinement(h, start_time, max_time):
    """Local refinement using FFT-accelerated sensitivity-based peak reduction.
    
    At each step, we identify lags where the autoconvolution is near its max,
    compute how much each bin contributes to those peaks via FFT, and reduce bins
    that contribute more to peaks than to off-peak regions.
    
    The FFT trick: sensitivity[j] = sum_{m: peak} h[m-j]
    This is a cross-correlation of h with the peak_mask, which can be computed
    as conv(h, flip(peak_mask)) and extracting the right slice.
    """
    best_h = h.copy()
    best_val = compute_score(h)
    current_h = h.copy()
    alpha = 0.03
    threshold_frac = 0.98
    min_alpha = 1e-7
    
    while time.time() - start_time < max_time:
        s = np.sum(current_h)
        if s < 1e-12:
            break
        # Normalize to sum = N for stable computation
        current_h = current_h * (N / s)
        
        conv = fft_convolve(current_h, current_h)
        max_c = np.max(conv)
        current_val = max_c / (N * N)
        
        if current_val < best_val:
            best_val = current_val
            best_h = current_h.copy()
        
        # Find lags where conv is close to max
        threshold = max_c * threshold_frac
        peak_mask = conv >= threshold
        n_peaks = np.sum(peak_mask)
        
        if n_peaks == 0:
            break
        
        # Compute sensitivity via FFT:
        # sensitivity[j] = sum_m peak_mask[m] * h[m-j]
        # This is cross-correlation, which equals conv(h, flip(peak_mask))
        peak_rev = peak_mask[::-1].astype(np.float64)
        sens_full = fft_convolve(current_h, peak_rev)
        # The cross-correlation at lag j is at index j + (N-1) in the full conv
        sens_offset = N - 1
        sensitivity = sens_full[sens_offset:sens_offset + N]
        
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
        
        conv_new = fft_convolve(h_new, h_new)
        val_new = np.max(conv_new) / (N * N)
        
        if val_new < current_val:
            current_h = h_new
            alpha = min(alpha * 1.02, 0.3)
        else:
            alpha *= 0.9
            if alpha < min_alpha:
                break
    
    return best_h, best_val

def spectral_refinement(h, start_time, max_time):
    """Spectral flattening: reduce bins that contribute to high-frequency peaks
    in the power spectrum to flatten the autoconvolution."""
    best_h = h.copy()
    best_val = compute_score(h)
    current_h = h.copy()
    alpha = 0.02
    
    while time.time() - start_time < max_time:
        s = np.sum(current_h)
        if s < 1e-12:
            break
        current_h = current_h * (N / s)
        
        conv = fft_convolve(current_h, current_h)
        max_c = np.max(conv)
        current_val = max_c / (N * N)
        
        if current_val < best_val:
            best_val = current_val
            best_h = current_h.copy()
        
        # Compute power spectrum
        F = np.fft.rfft(current_h, N)
        power = np.abs(F) ** 2
        power_max = np.max(power)
        
        if power_max < 1e-15:
            break
        
        # Identify high-power frequency bins
        threshold = power_max * 0.85
        high_power_mask = power >= threshold
        
        # Compute which time-domain bins contribute to high-power frequencies
        # |F[k]|^2 = |sum_j h[j] exp(-2pi i k j / N)|^2
        # Gradient w.r.t. h[j] is complex; use real part of inverse FFT of
        # high_power_mask * F
        F_complex = np.fft.rfft(current_h, N)
        mask_complex = np.zeros_like(F_complex)
        mask_complex[:len(high_power_mask)] = high_power_mask[:len(F_complex)]
        
        # Gradient of power w.r.t. h is related to IFFT(mask * F)
        grad = np.fft.irfft(mask_complex * F_complex, N)
        
        # Normalize gradient
        grad_mean = np.mean(grad)
        grad_std = np.std(grad) + 1e-12
        grad_norm = (grad - grad_mean) / grad_std
        
        # Reduce bins with high positive gradient
        h_new = current_h * (1.0 - alpha * np.maximum(grad_norm, 0))
        h_new = np.maximum(h_new, 0.0)
        
        s_new = np.sum(h_new)
        if s_new < 1e-12:
            break
        h_new = h_new * (N / s_new)
        
        conv_new = fft_convolve(h_new, h_new)
        val_new = np.max(conv_new) / (N * N)
        
        if val_new < current_val:
            current_h = h_new
            alpha = min(alpha * 1.02, 0.2)
        else:
            alpha *= 0.9
            if alpha < 1e-8:
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
            if time.time() - start_time > TIME_LIMIT - 25:
                break
            
            for restart in range(6):
                if time.time() - start_time > TIME_LIMIT - 20:
                    break
                
                centers = rng.uniform(-0.2, 0.2, size=n_c)
                widths = rng.uniform(0.05, 0.25, size=n_c)
                heights = rng.uniform(0.3, 3.0, size=n_c)
                
                x0 = np.concatenate([centers, widths, heights])
                
                try:
                    result = minimize(
                        objective, x0, args=(n_c, beta),
                        method='Nelder-Mead',
                        options={'maxiter': 2000, 'xatol': 1e-7, 'fatol': 1e-9}
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
    
    # Local refinement: FFT-accelerated sensitivity-based peak reduction
    if time.time() - start_time < TIME_LIMIT - 20:
        refined_f, refined_score = local_refinement(
            best_f, start_time, TIME_LIMIT - 10
        )
        if refined_score < best_score:
            best_f = refined_f
            best_score = refined_score
    
    # Spectral refinement pass
    if time.time() - start_time < TIME_LIMIT - 10:
        refined_f2, refined_score2 = spectral_refinement(
            best_f, start_time, TIME_LIMIT - 5
        )
        if refined_score2 < best_score:
            best_f = refined_f2
            best_score = refined_score2
    
    # Second local refinement pass
    if time.time() - start_time < TIME_LIMIT - 5:
        refined_f3, refined_score3 = local_refinement(
            best_f, start_time, TIME_LIMIT - 2
        )
        if refined_score3 < best_score:
            best_f = refined_f3
            best_score = refined_score3
    
    # Ensure nonnegative
    best_f = np.maximum(best_f, 0.0)
    return [float(x) for x in best_f]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
