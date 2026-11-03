"""
First autocorrelation inequality: minimize normalized max of autoconvolution
for a nonnegative step function with 600 equal intervals on [-1/4, 1/4].

Strategy:
1. Initialize with smooth Gaussian bump family.
2. Use frequency-domain coordinate descent to reduce autoconvolution peak.
3. Apply local refinement using sensitivity-based peak reduction with FFT.
4. Add a final polish pass with adaptive step sizes.

Key improvements:
- Frequency-domain optimization leveraging Parseval's theorem
- FFT-based autoconvolution for O(N log N)
- More aggressive local refinement with adaptive thresholds
- Better initialization from multiple random restarts
"""

import numpy as np
from scipy.signal import fftconvolve
import time

N = 600
TIME_LIMIT = 140  # safety margin before 150s kill

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

def freq_domain_optimize(f, start_time, max_time, n_sweeps=10):
    """
    Optimize in frequency domain using coordinate descent.
    For autoconvolution, the peak relates to the L2 norm of the Fourier transform.
    We minimize the time-domain peak by adjusting frequency components.
    """
    best_f = f.copy()
    best_val = compute_score(f)
    current_f = f.copy()
    
    # Normalize
    s = np.sum(current_f)
    if s > 0:
        current_f = current_f * (N / s)
    
    # Get FFT
    F = np.fft.fft(current_f)
    F_abs = np.abs(F)
    
    # We'll do coordinate descent on the magnitude spectrum
    # The autoconvolution in frequency domain is |F|^2 (up to constants)
    # Minimizing the peak of ifft(|F|^2) is equivalent to our objective
    
    step_sizes = [0.05, 0.03, 0.02, 0.01, 0.005, 0.003, 0.002, 0.001]
    
    for sweep in range(n_sweeps):
        if time.time() - start_time > max_time:
            break
        
        ss = step_sizes[min(sweep, len(step_sizes)-1)]
        improved = False
        
        for freq_idx in range(N // 2 + 1):
            if time.time() - start_time > max_time:
                break
            
            # Try reducing this frequency component
            old_mag = F_abs[freq_idx].copy()
            
            # For DC (freq 0), be careful
            if freq_idx == 0:
                new_mag = old_mag * (1 - ss * 0.1)  # Small reduction
            else:
                new_mag = old_mag * (1 - ss)
            
            if new_mag < 0:
                new_mag = 0
            
            # Reconstruct
            F_new = F.copy()
            F_new[freq_idx] = new_mag * (F[freq_idx] / (old_mag + 1e-30))
            if freq_idx > 0 and freq_idx < N:
                F_new[N - freq_idx] = new_mag * (F[N - freq_idx] / (old_mag + 1e-30))
            
            f_new = np.real(np.fft.ifft(F_new))
            f_new = np.maximum(f_new, 0.0)
            
            s_new = np.sum(f_new)
            if s_new < 1e-12:
                continue
            f_new = f_new * (N / s_new)
            
            conv_new = fftconvolve(f_new, f_new, mode='full')
            val_new = np.max(conv_new) / (N * N)
            
            if val_new < best_val - 1e-10:
                current_f = f_new
                F = F_new
                F_abs[freq_idx] = new_mag
                if freq_idx > 0 and freq_idx < N:
                    F_abs[N - freq_idx] = new_mag
                improved = True
        
        if not improved:
            break
    
    final_val = compute_score(current_f)
    if final_val < best_val:
        best_f = current_f
        best_val = final_val
    
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
        # For each bin j, sensitivity = sum over peak lags m of h[m-j]
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
    best_score = compute_score(best_f)
    
    # Try different numbers of Gaussian bump components for initialization
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25 / N
    
    for n_c in [2, 3, 4, 5, 6]:
        if time.time() - start_time > TIME_LIMIT - 60:
            break
        
        rng = np.random.RandomState(42 + n_c)
        
        for restart in range(8):
            if time.time() - start_time > TIME_LIMIT - 55:
                break
            
            # Structured initialization
            if restart == 0:
                # Symmetric initialization
                centers = np.linspace(-0.15, 0.15, n_c)
                widths = np.full(n_c, 0.12)
                heights = np.full(n_c, 1.0)
            elif restart == 1:
                # Asymmetric with varying widths
                centers = rng.uniform(-0.2, 0.2, size=n_c)
                widths = rng.uniform(0.04, 0.2, size=n_c)
                heights = rng.uniform(0.5, 2.0, size=n_c)
            elif restart == 2:
                # Concentrated at center
                centers = np.zeros(n_c) + rng.uniform(-0.05, 0.05, size=n_c)
                widths = rng.uniform(0.03, 0.1, size=n_c)
                heights = rng.uniform(0.8, 2.5, size=n_c)
            elif restart == 3:
                # Wide spread
                centers = rng.uniform(-0.22, 0.22, size=n_c)
                widths = rng.uniform(0.1, 0.3, size=n_c)
                heights = rng.uniform(0.3, 1.5, size=n_c)
            else:
                centers = rng.uniform(-0.2, 0.2, size=n_c)
                widths = rng.uniform(0.02, 0.25, size=n_c)
                heights = rng.uniform(0.3, 3.0, size=n_c)
            
            f = gaussian_basis(x, centers, widths, heights)
            f = np.maximum(f, 0.0)
            
            # Normalize
            s = np.sum(f)
            if s < 1e-12:
                continue
            f = f * (N / s)
            
            score = compute_score(f)
            if score < best_score:
                best_f = f.copy()
                best_score = score
    
    # Frequency domain optimization
    if time.time() - start_time < TIME_LIMIT - 30:
        freq_f, freq_score = freq_domain_optimize(
            best_f, start_time, TIME_LIMIT - 15
        )
        if freq_score < best_score:
            best_f = freq_f
            best_score = freq_score
    
    # Local refinement: sensitivity-based peak reduction
    if time.time() - start_time < TIME_LIMIT - 20:
        refined_f, refined_score = local_refinement(
            best_f, start_time, TIME_LIMIT - 10
        )
        if refined_score < best_score:
            best_f = refined_f
            best_score = refined_score
    
    # Final polish pass
    if time.time() - start_time < TIME_LIMIT - 5:
        polished_f, polished_score = polish_pass(
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
