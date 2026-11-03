"""
First autocorrelation inequality: minimize normalized max of autoconvolution
for a nonnegative step function with 600 equal intervals on [-1/4, 1/4].

Strategy: Use a smooth parametric family based on cosine bumps, then refine
with a spectral-domain projected gradient descent that directly optimizes
the frequency response to flatten the autoconvolution peak.

The key insight is to work in the frequency domain where the autoconvolution
is the power spectrum. We use projected gradient descent on the frequency
domain with soft-thresholding to enforce non-negativity of the time-domain
signal, combined with a sensitivity-based local refinement.
"""

import numpy as np
from scipy.optimize import minimize
import time

N = 600
TIME_LIMIT = 130  # safety margin before 150s kill

def compute_score(f):
    """Compute the normalized maximum of the autoconvolution."""
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return np.inf
    f = np.maximum(f, 0.0)
    conv = np.convolve(f, f, mode='full')
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
    conv = np.convolve(f, f, mode='full')
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    return smooth_max(conv, beta) / (s * s)

def spectral_projected_gradient(h, start_time, max_time):
    """Spectral-domain projected gradient descent with non-negativity constraint.
    
    Works in frequency domain to directly optimize the power spectrum,
    using projected gradient descent with soft-thresholding to enforce
    non-negativity of the time-domain signal.
    """
    best_h = h.copy()
    best_val = compute_score(h)
    current_h = h.copy()
    n_fft = 2 * N
    
    # Initialize step size and learning rate
    lr = 0.01
    min_lr = 1e-6
    max_lr = 0.1
    threshold_frac = 0.95
    
    # Precompute FFT of ones for normalization
    ones_fft = np.fft.rfft(np.ones(N), n=n_fft)
    
    while time.time() - start_time < max_time:
        s = np.sum(current_h)
        if s < 1e-12:
            break
        # Normalize to sum = N for stable computation
        current_h = current_h * (N / s)
        
        # Use FFT for fast convolution
        f_hat = np.fft.rfft(current_h, n=n_fft)
        conv = np.fft.irfft(f_hat * f_hat, n=n_fft)[:2*N-1]
        max_c = np.max(conv)
        current_val = max_c / (N * N)
        
        if current_val < best_val:
            best_val = current_val
            best_h = current_h.copy()
        
        # Find lags where conv is close to max
        threshold = max_c * threshold_frac
        peak_mask = (conv >= threshold).astype(np.float64)
        
        # Compute gradient in frequency domain
        # The objective is max(conv) / (sum(f))^2
        # Gradient w.r.t. f_hat: we want to reduce power at frequencies
        # that contribute to peak lags
        
        # Compute sensitivity via FFT
        pm_hat = np.fft.rfft(peak_mask, n=n_fft)
        ch_hat = np.fft.rfft(current_h, n=n_fft)
        # Cross-correlation: ifft(fft(pm) * conj(fft(ch)))
        sens = np.fft.irfft(pm_hat * np.conj(ch_hat), n=n_fft)[:N]
        
        # Normalize sensitivity
        sens_mean = np.mean(sens)
        sens_std = np.std(sens) + 1e-12
        sens_norm = (sens - sens_mean) / sens_std
        
        # Compute gradient in frequency domain
        # grad_f_hat = 2 * conj(f_hat) * sens_hat (for the peak reduction)
        sens_hat = np.fft.rfft(sens_norm, n=n_fft)
        grad_f_hat = 2.0 * np.conj(f_hat) * sens_hat
        
        # Project gradient descent step in frequency domain
        f_hat_new = f_hat - lr * grad_f_hat
        
        # Inverse FFT to get time domain
        h_new = np.fft.irfft(f_hat_new, n=n_fft)[:N]
        
        # Enforce non-negativity via soft-thresholding
        h_new = np.maximum(h_new, 0.0)
        
        # Normalize to maintain sum = N
        s_new = np.sum(h_new)
        if s_new < 1e-12:
            break
        h_new = h_new * (N / s_new)
        
        # Evaluate new candidate
        f_new_hat = np.fft.rfft(h_new, n=n_fft)
        conv_new = np.fft.irfft(f_new_hat * f_new_hat, n=n_fft)[:2*N-1]
        val_new = np.max(conv_new) / (N * N)
        
        # Adaptive step size
        if val_new < current_val:
            current_h = h_new
            lr = min(lr * 1.1, max_lr)
        else:
            lr = max(lr * 0.8, min_lr)
            if lr < min_lr * 1.1:
                # Try smaller step
                f_hat_new = f_hat - (lr * 0.5) * grad_f_hat
                h_new = np.fft.irfft(f_hat_new, n=n_fft)[:N]
                h_new = np.maximum(h_new, 0.0)
                s_new = np.sum(h_new)
                if s_new < 1e-12:
                    break
                h_new = h_new * (N / s_new)
                f_new_hat = np.fft.rfft(h_new, n=n_fft)
                conv_new = np.fft.irfft(f_new_hat * f_new_hat, n=n_fft)[:2*N-1]
                val_new = np.max(conv_new) / (N * N)
                if val_new < current_val:
                    current_h = h_new
                else:
                    break
    
    return best_h, best_val

def local_refinement(h, start_time, max_time):
    """Local refinement using FFT-based sensitivity peak reduction.
    
    Uses FFT to efficiently compute sensitivity of each bin to peak lags,
    then reduces bins that disproportionately contribute to the maximum
    autoconvolution values.
    """
    best_h = h.copy()
    best_val = compute_score(h)
    current_h = h.copy()
    alpha = 0.05
    threshold_frac = 0.95
    n_fft = 2 * N  # size for full convolution via FFT
    
    while time.time() - start_time < max_time:
        s = np.sum(current_h)
        if s < 1e-12:
            break
        # Normalize to sum = N for stable computation
        current_h = current_h * (N / s)
        
        # Use FFT for fast convolution
        f_hat = np.fft.rfft(current_h, n=n_fft)
        conv = np.fft.irfft(f_hat * f_hat, n=n_fft)[:2*N-1]
        max_c = np.max(conv)
        current_val = max_c / (N * N)
        
        if current_val < best_val:
            best_val = current_val
            best_h = current_h.copy()
        
        # Find lags where conv is close to max
        threshold = max_c * threshold_frac
        peak_mask = (conv >= threshold).astype(np.float64)
        
        # Compute sensitivity via FFT:
        # sensitivity[j] = sum_m peak_mask[m] * current_h[m-j] (where valid)
        # This is a cross-correlation of peak_mask with current_h
        # Use FFT: sensitivity = ifft(fft(peak_mask) * conj(fft(current_h_padded)))
        pm_hat = np.fft.rfft(peak_mask, n=n_fft)
        ch_hat = np.fft.rfft(current_h, n=n_fft)
        # Cross-correlation: ifft(fft(pm) * conj(fft(ch)))
        sens = np.fft.irfft(pm_hat * np.conj(ch_hat), n=n_fft)[:N]
        
        # Normalize sensitivity
        sens_mean = np.mean(sens)
        sens_std = np.std(sens) + 1e-12
        sens_norm = (sens - sens_mean) / sens_std
        
        # Reduce bins with high sensitivity (they contribute to peaks)
        h_new = current_h * (1.0 - alpha * sens_norm)
        h_new = np.maximum(h_new, 0.0)
        
        s_new = np.sum(h_new)
        if s_new < 1e-12:
            break
        h_new = h_new * (N / s_new)
        
        # Evaluate new candidate
        f_new_hat = np.fft.rfft(h_new, n=n_fft)
        conv_new = np.fft.irfft(f_new_hat * f_new_hat, n=n_fft)[:2*N-1]
        val_new = np.max(conv_new) / (N * N)
        
        if val_new < current_val:
            current_h = h_new
            alpha = min(alpha * 1.05, 0.5)
        else:
            alpha *= 0.85
            if alpha < 1e-8:
                break
    
    return best_h, best_val

def construct():
    start_time = time.time()
    
    # Initial best: constant function
    best_f = np.ones(N)
    best_score = compute_score(best_f)
    
    # Try different numbers of cosine bump components with tighter time budget
    # Reserve at least 60% of time for local refinement
    param_budget_end = start_time + (TIME_LIMIT - 5) * 0.35
    
    for n_c in [2, 3, 4]:
        if time.time() > param_budget_end:
            break
        
        rng = np.random.RandomState(42 + n_c)
        best_params = None
        best_obj = np.inf
        
        # Fewer restarts, tighter tolerances to save time
        betas = [10.0, 20.0]
        
        for beta in betas:
            if time.time() > param_budget_end:
                break
            
            for restart in range(3):
                if time.time() > param_budget_end:
                    break
                
                centers = rng.uniform(-0.15, 0.15, size=n_c)
                widths = rng.uniform(0.08, 0.25, size=n_c)
                heights = rng.uniform(0.5, 2.0, size=n_c)
                
                x0 = np.concatenate([centers, widths, heights])
                
                try:
                    result = minimize(
                        objective, x0, args=(n_c, beta),
                        method='Nelder-Mead',
                        options={'maxiter': 800, 'xatol': 1e-5, 'fatol': 1e-7}
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
    
    # Local refinement: FFT-based sensitivity peak reduction
    # This is now fast enough to run for most of the remaining time
    if time.time() < TIME_LIMIT - 5:
        refined_f, refined_score = local_refinement(
            best_f, start_time, TIME_LIMIT - 5
        )
        if refined_score < best_score:
            best_f = refined_f
            best_score = refined_score
    
    # Spectral projected gradient descent
    if time.time() < TIME_LIMIT - 3:
        spectral_f, spectral_score = spectral_projected_gradient(
            best_f, start_time, TIME_LIMIT - 3
        )
        if spectral_score < best_score:
            best_f = spectral_f
            best_score = spectral_score
    
    # Final polish: one more refinement pass with smaller alpha
    if time.time() < TIME_LIMIT - 2:
        best_f = best_f.copy()
        alpha_small = 0.02
        for _ in range(50):
            if time.time() > TIME_LIMIT - 2:
                break
            s = np.sum(best_f)
            if s < 1e-12:
                break
            best_f = best_f * (N / s)
            n_fft = 2 * N
            f_hat = np.fft.rfft(best_f, n=n_fft)
            conv = np.fft.irfft(f_hat * f_hat, n=n_fft)[:2*N-1]
            max_c = np.max(conv)
            current_val = max_c / (N * N)
            
            threshold = max_c * 0.98
            peak_mask = (conv >= threshold).astype(np.float64)
            pm_hat = np.fft.rfft(peak_mask, n=n_fft)
            ch_hat = np.fft.rfft(best_f, n=n_fft)
            sens = np.fft.irfft(pm_hat * np.conj(ch_hat), n=n_fft)[:N]
            
            sens_mean = np.mean(sens)
            sens_std = np.std(sens) + 1e-12
            sens_norm = (sens - sens_mean) / sens_std
            
            h_new = best_f * (1.0 - alpha_small * sens_norm)
            h_new = np.maximum(h_new, 0.0)
            s_new = np.sum(h_new)
            if s_new < 1e-12:
                break
            h_new = h_new * (N / s_new)
            
            f_new_hat = np.fft.rfft(h_new, n=n_fft)
            conv_new = np.fft.irfft(f_new_hat * f_new_hat, n=n_fft)[:2*N-1]
            val_new = np.max(conv_new) / (N * N)
            
            if val_new < current_val:
                best_f = h_new
            else:
                break
    
    # Ensure nonnegative
    best_f = np.maximum(best_f, 0.0)
    return [float(x) for x in best_f]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
