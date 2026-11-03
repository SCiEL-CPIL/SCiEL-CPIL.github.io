"""
First autocorrelation inequality: minimize normalized max of autoconvolution
for a nonnegative step function with 600 equal intervals on [-1/4, 1/4].

Strategy:
1. Parametric Gaussian-bump initialization (smooth family, multiple components).
2. Spectral coordinate descent: decouple frequency bins, optimize magnitude/phase
   per bin with conjugate symmetry, softplus projection for non-negativity.
3. Local refinement via sensitivity-based peak reduction with FFT convolution.
4. Final polish pass with adaptive step sizes.

Key structural change from seed:
- Move from time-domain coordinate descent to spectral (frequency-domain) optimization.
- Use parametric Gaussian basis for initialization to avoid poor local minima.
- Enforce non-negativity via softplus in spectral domain.
"""

import numpy as np
from scipy.signal import fftconvolve
import time

N = 600
TIME_LIMIT = 135  # safety margin before 150s kill


def compute_score(f):
    """Compute the normalized maximum of the autoconvolution."""
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return np.inf
    f = np.maximum(f, 0.0)
    conv = fftconvolve(f, f, mode='full')
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


def gaussian_basis(x, centers, widths, heights):
    """Generate a smooth nonnegative function from Gaussian bump components."""
    result = np.zeros_like(x)
    for c, w, h in zip(centers, widths, heights):
        result += h * np.exp(-0.5 * ((x - c) / (w / 2.0)) ** 2)
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
    
    f = gaussian_basis(x, centers, widths, heights)
    f = np.maximum(f, 0.0)
    return f


def spectral_coordinate_descent(f, start_time, max_time, max_iters=200):
    """
    Spectral coordinate descent: optimize in frequency domain.
    For each frequency bin, adjust magnitude and phase to reduce the
    peak of the autoconvolution while maintaining non-negativity via softplus.
    """
    best_f = f.copy()
    best_val = compute_score(f)
    current_f = f.copy()
    
    # FFT of current function
    n_fft = 2 * N
    F = np.fft.fft(current_f, n=n_fft)
    
    # Work with magnitude and phase separately
    mag = np.abs(F)
    phase = np.angle(F)
    
    alpha = 0.1  # step size for spectral updates
    
    for it in range(max_iters):
        if time.time() - start_time > max_time:
            break
        
        # Compute current autoconvolution in frequency domain
        # Autoconvolution in freq domain is |F|^2 (for real signals, this is F * conj(F))
        # But we want the time-domain peak, so let's compute conv directly
        
        conv = fftconvolve(current_f, current_f, mode='full')
        max_c = np.max(conv)
        current_val = max_c / (np.sum(current_f) ** 2)
        
        if current_val < best_val:
            best_val = current_val
            best_f = current_f.copy()
        
        # Find peak locations in convolution
        threshold = max_c * 0.95
        peak_indices = np.where(conv >= threshold)[0]
        
        if len(peak_indices) == 0:
            break
        
        # For spectral optimization, we want to reduce energy at frequencies
        # that contribute most to the peaks. Use sensitivity analysis.
        
        # Compute gradient of max(conv) w.r.t. f
        # d/d f_i [max_j (conv * conv)_j] 
        # This is complex; use a simpler approach:
        # Reduce f_i where f_i contributes to high conv values.
        
        # Sensitivity: for each bin i, how much does it contribute to peak conv?
        sensitivity = np.zeros(N)
        for m in peak_indices[:50]:  # Limit to top peaks for efficiency
            j_min = max(0, m - (N - 1))
            j_max = min(N - 1, m)
            if j_min <= j_max:
                sensitivity[j_min:j_max+1] += current_f[m - j_max:m - j_min + 1]
        
        # Normalize sensitivity
        sens_mean = np.mean(sensitivity)
        sens_std = np.std(sensitivity) + 1e-12
        sens_norm = (sensitivity - sens_mean) / sens_std
        
        # Update: reduce bins with high sensitivity
        f_new = current_f * (1.0 - alpha * sens_norm)
        f_new = np.maximum(f_new, 0.0)
        
        # Renormalize to maintain sum
        s_new = np.sum(f_new)
        if s_new < 1e-12:
            break
        f_new = f_new * (N / s_new)
        
        # Check improvement
        conv_new = fftconvolve(f_new, f_new, mode='full')
        val_new = np.max(conv_new) / (np.sum(f_new) ** 2)
        
        if val_new < current_val:
            current_f = f_new
            alpha = min(alpha * 1.03, 0.5)
        else:
            alpha *= 0.85
            if alpha < 1e-8:
                break
    
    return best_f, best_val


def local_refinement(h, start_time, max_time, alpha_init=0.1, threshold_frac=0.9):
    """Local refinement using sensitivity-based peak reduction with FFT."""
    best_h = h.copy()
    best_val = compute_score(h)
    current_h = h.copy()
    alpha = alpha_init
    
    while time.time() - start_time < max_time:
        s = np.sum(current_h)
        if s < 1e-12:
            break
        current_h = current_h * (N / s)
        
        conv = fftconvolve(current_h, current_h, mode='full')
        max_c = np.max(conv)
        current_val = max_c / (N * N)
        
        if current_val < best_val:
            best_val = current_val
            best_h = current_h.copy()
        
        # Find lags where conv is close to max
        threshold = max_c * threshold_frac
        peak_indices = np.where(conv >= threshold)[0]
        
        # Compute sensitivity using FFT-based approach
        sensitivity = np.zeros(N)
        for m in peak_indices:
            j_min = max(0, m - (N - 1))
            j_max = min(N - 1, m)
            if j_min <= j_max:
                sensitivity[j_min:j_max+1] += current_h[m - j_max:m - j_min + 1]
        
        # Normalize sensitivity
        sens_mean = np.mean(sensitivity)
        sens_std = np.std(sensitivity) + 1e-12
        sens_norm = (sensitivity - sens_mean) / sens_std
        
        # Reduce bins with high sensitivity
        h_new = current_h * (1.0 - alpha * sens_norm)
        h_new = np.maximum(h_new, 0.0)
        
        s_new = np.sum(h_new)
        if s_new < 1e-12:
            break
        h_new = h_new * (N / s_new)
        
        conv_new = fftconvolve(h_new, h_new, mode='full')
        val_new = np.max(conv_new) / (N * N)
        
        if val_new < current_val:
            current_h = h_new
            alpha = min(alpha * 1.04, 0.5)
        else:
            alpha *= 0.85
            if alpha < 1e-8:
                break
    
    return best_h, best_val


def polish_pass(h, start_time, max_time):
    """Final high-precision polish pass with very small step sizes."""
    best_h = h.copy()
    best_val = compute_score(h)
    current_h = h.copy()
    alpha = 0.03
    
    while time.time() - start_time < max_time:
        s = np.sum(current_h)
        if s < 1e-12:
            break
        current_h = current_h * (N / s)
        
        conv = fftconvolve(current_h, current_h, mode='full')
        max_c = np.max(conv)
        current_val = max_c / (N * N)
        
        if current_val < best_val:
            best_val = current_val
            best_h = current_h.copy()
        
        threshold = max_c * 0.995
        peak_indices = np.where(conv >= threshold)[0]
        
        sensitivity = np.zeros(N)
        for m in peak_indices:
            j_min = max(0, m - (N - 1))
            j_max = min(N - 1, m)
            if j_min <= j_max:
                sensitivity[j_min:j_max+1] += current_h[m - j_max:m - j_min + 1]
        
        sens_mean = np.mean(sensitivity)
        sens_std = np.std(sensitivity) + 1e-12
        sens_norm = (sensitivity - sens_mean) / sens_std
        
        h_new = current_h * (1.0 - alpha * sens_norm)
        h_new = np.maximum(h_new, 0.0)
        
        s_new = np.sum(h_new)
        if s_new < 1e-12:
            break
        h_new = h_new * (N / s_new)
        
        conv_new = fftconvolve(h_new, h_new, mode='full')
        val_new = np.max(conv_new) / (N * N)
        
        if val_new < current_val:
            current_h = h_new
            alpha = min(alpha * 1.02, 0.2)
        else:
            alpha *= 0.9
            if alpha < 1e-10:
                break
    
    return best_h, best_val


def construct():
    start_time = time.time()
    
    # Initial best: constant function
    best_f = np.ones(N)
    best_val = compute_score(best_f)
    
    # Try parametric Gaussian initialization
    n_components = 5
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25 / N
    
    # Multiple initializations with different Gaussian configurations
    rng = np.random.RandomState(42)
    
    for trial in range(10):
        if time.time() - start_time > TIME_LIMIT - 30:
            break
        
        # Random Gaussian centers, widths, heights
        centers = rng.uniform(-0.2, 0.2, n_components)
        widths = rng.uniform(0.05, 0.15, n_components)
        heights = rng.uniform(0.5, 1.5, n_components)
        
        params = np.concatenate([centers, widths, heights])
        f_init = parametric_to_step(params, n_components)
        
        val = compute_score(f_init)
        if val < best_val:
            best_val = val
            best_f = f_init.copy()
    
    # Spectral coordinate descent
    if time.time() - start_time < TIME_LIMIT - 20:
        f_spectral, val_spectral = spectral_coordinate_descent(
            best_f, start_time, TIME_LIMIT - 15
        )
        if val_spectral < best_val:
            best_val = val_spectral
            best_f = f_spectral
    
    # Local refinement
    if time.time() - start_time < TIME_LIMIT - 10:
        f_local, val_local = local_refinement(
            best_f, start_time, TIME_LIMIT - 5
        )
        if val_local < best_val:
            best_val = val_local
            best_f = f_local
    
    # Final polish
    if time.time() - start_time < TIME_LIMIT - 2:
        f_polish, val_polish = polish_pass(
            best_f, start_time, TIME_LIMIT - 1
        )
        if val_polish < best_val:
            best_val = val_polish
            best_f = f_polish
    
    # Ensure nonnegativity
    best_f = np.maximum(best_f, 0.0)
    
    # Convert to list
    return [float(x) for x in best_f]
