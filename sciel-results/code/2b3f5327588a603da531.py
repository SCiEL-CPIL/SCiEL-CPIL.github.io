"""
First autocorrelation inequality: minimize normalized max of autoconvolution
for a nonnegative step function with 600 equal intervals on [-1/4, 1/4].

Strategy: Use a smooth parametric family based on cosine bumps, then refine
with a sensitivity-based local optimization that targets peak reduction.

The key insight is to use a family of functions whose autoconvolution has a
flatter peak. We use a smooth max (log-sum-exp) surrogate for the parametric
optimization, then apply a gradient-like local refinement that reduces the
sensitivity of peak lags to reduce the maximum autoconvolution value.
"""

import numpy as np
from scipy.optimize import minimize
import time

N = 600
TIME_LIMIT = 145  # safety margin before 150s kill

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

def complex_gaussian_basis(x, centers, widths, freqs, real_coeffs, imag_coeffs):
    """Generate |z(x)| where z(x) = sum_k (a_k + i*b_k) * exp(-w_k*(x-c_k)^2) * exp(i*f_k*x).
    
    Parameters:
        x: array of grid points
        centers: array of mode centers
        widths: array of mode widths (positive)
        freqs: array of mode frequencies
        real_coeffs: array of real parts of complex coefficients
        imag_coeffs: array of imaginary parts of complex coefficients
    
    Returns:
        f: array of |z(x)| values (nonnegative)
    """
    n_modes = len(centers)
    z = np.zeros_like(x, dtype=np.complex128)
    for k in range(n_modes):
        coeff = real_coeffs[k] + 1j * imag_coeffs[k]
        gaussian = np.exp(-widths[k] * (x - centers[k]) ** 2)
        phase = np.exp(1j * freqs[k] * x)
        z += coeff * gaussian * phase
    return np.abs(z)

def parametric_to_step(params, n_components):
    """Convert parameters to a step function of length N.
    
    Parameters are ordered as:
    [centers (n), widths (n), freqs (n), real_coeffs (n), imag_coeffs (n)]
    Total: 5*n parameters
    """
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25 / N
    centers = params[:n_components]
    widths = np.abs(params[n_components:2*n_components]) + 0.05
    widths = np.minimum(widths, 50.0)  # cap to avoid too narrow modes
    freqs = params[2*n_components:3*n_components]
    real_coeffs = params[3*n_components:4*n_components]
    imag_coeffs = params[4*n_components:5*n_components]
    
    f = complex_gaussian_basis(x, centers, widths, freqs, real_coeffs, imag_coeffs)
    f = np.maximum(f, 0.0)
    return f

def objective(params, n_components, beta=10.0):
    f = parametric_to_step(params, n_components)
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    # Use FFT for faster convolution
    n_fft = 2 * N
    f_hat = np.fft.rfft(f, n=n_fft)
    conv = np.fft.irfft(f_hat * f_hat, n=n_fft)[:2*N-1]
    return smooth_max(conv, beta) / (s * s)

def local_refinement(h, start_time, max_time):
    """Local refinement using FFT-based sensitivity peak reduction with adaptive
    threshold, smooth peak mask, and backtracking line search."""
    best_h = h.copy()
    best_val = compute_score(h)
    current_h = h.copy()
    alpha = 0.03
    threshold_frac = 0.97
    n_fft = 2 * N  # size for full convolution via FFT
    
    # Track convergence
    no_improve_count = 0
    max_no_improve = 15
    
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
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        if no_improve_count > max_no_improve:
            break
        
        # Find lags where conv is close to max - use smooth peak mask
        # instead of hard binary mask to get smoother gradient
        threshold = max_c * threshold_frac
        # Smooth mask: linearly ramp from 0 to 1 between threshold*0.95 and threshold
        lower_thresh = max_c * 0.95
        peak_mask = np.clip((conv - lower_thresh) / (threshold - lower_thresh + 1e-15), 0, 1)
        
        # Compute sensitivity via FFT:
        # sensitivity[j] = sum_m peak_mask[m] * current_h[m-j] (where valid)
        # This is a cross-correlation of peak_mask with current_h
        pm_hat = np.fft.rfft(peak_mask, n=n_fft)
        ch_hat = np.fft.rfft(current_h, n=n_fft)
        # Cross-correlation: ifft(fft(pm) * conj(fft(ch)))
        sens = np.fft.irfft(pm_hat * np.conj(ch_hat), n=n_fft)[:N]
        
        # Normalize sensitivity
        sens_mean = np.mean(sens)
        sens_std = np.std(sens) + 1e-12
        sens_norm = (sens - sens_mean) / sens_std
        
        # Backtracking line search to find best step size
        best_alpha = alpha
        best_h_candidate = None
        best_val_candidate = current_val
        
        for trial_alpha in [alpha, alpha * 0.5, alpha * 2.0, alpha * 0.25, alpha * 4.0]:
            h_new = current_h * (1.0 - trial_alpha * sens_norm)
            h_new = np.maximum(h_new, 0.0)
            
            s_new = np.sum(h_new)
            if s_new < 1e-12:
                continue
            h_new = h_new * (N / s_new)
            
            # Evaluate new candidate
            f_new_hat = np.fft.rfft(h_new, n=n_fft)
            conv_new = np.fft.irfft(f_new_hat * f_new_hat, n=n_fft)[:2*N-1]
            val_new = np.max(conv_new) / (N * N)
            
            if val_new < best_val_candidate:
                best_val_candidate = val_new
                best_alpha = trial_alpha
                best_h_candidate = h_new
        
        if best_h_candidate is not None and best_val_candidate < current_val:
            current_h = best_h_candidate
            # Adapt alpha based on what worked
            if best_alpha > alpha:
                alpha = min(alpha * 1.1, 0.2)
            else:
                alpha = max(alpha * 0.9, 0.01)
        else:
            # No improvement found, reduce alpha
            alpha *= 0.7
            if alpha < 1e-6:
                break
    
    return best_h, best_val

def construct():
    start_time = time.time()
    
    # Initial best: constant function
    best_f = np.ones(N)
    best_score = compute_score(best_f)
    
    # Try different numbers of complex Gaussian modes with tighter time budget
    # Reserve at least 65% of time for local refinement
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
                widths = rng.uniform(2.0, 20.0, size=n_c)  # width parameter for exp(-w*x^2)
                freqs = rng.uniform(-5.0, 5.0, size=n_c)
                real_coeffs = rng.uniform(0.5, 2.0, size=n_c)
                imag_coeffs = rng.uniform(-1.0, 1.0, size=n_c)
                
                x0 = np.concatenate([centers, widths, freqs, real_coeffs, imag_coeffs])
                
                try:
                    result = minimize(
                        objective, x0, args=(n_c, beta),
                        method='Nelder-Mead',
                        options={'maxiter': 600, 'xatol': 1e-5, 'fatol': 1e-7}
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
    
    # Final polish: high-precision refinement with very small steps
    if time.time() < TIME_LIMIT - 2:
        best_f = best_f.copy()
        alpha_small = 0.005
        n_fft = 2 * N
        
        for _ in range(200):
            if time.time() > TIME_LIMIT - 2:
                break
            s = np.sum(best_f)
            if s < 1e-12:
                break
            best_f = best_f * (N / s)
            f_hat = np.fft.rfft(best_f, n=n_fft)
            conv = np.fft.irfft(f_hat * f_hat, n=n_fft)[:2*N-1]
            max_c = np.max(conv)
            current_val = max_c / (N * N)
            
            # Use smooth peak mask for tighter focus
            threshold = max_c * 0.998
            lower_thresh = max_c * 0.995
            peak_mask = np.clip((conv - lower_thresh) / (threshold - lower_thresh + 1e-15), 0, 1)
            
            pm_hat = np.fft.rfft(peak_mask, n=n_fft)
            ch_hat = np.fft.rfft(best_f, n=n_fft)
            sens = np.fft.irfft(pm_hat * np.conj(ch_hat), n=n_fft)[:N]
            
            sens_mean = np.mean(sens)
            sens_std = np.std(sens) + 1e-12
            sens_norm = (sens - sens_mean) / sens_std
            
            # Try multiple step sizes with backtracking
            best_alpha = alpha_small
            best_candidate = None
            best_val = current_val
            
            for trial_alpha in [alpha_small, alpha_small * 0.5, alpha_small * 2.0]:
                h_new = best_f * (1.0 - trial_alpha * sens_norm)
                h_new = np.maximum(h_new, 0.0)
                s_new = np.sum(h_new)
                if s_new < 1e-12:
                    continue
                h_new = h_new * (N / s_new)
                
                f_new_hat = np.fft.rfft(h_new, n=n_fft)
                conv_new = np.fft.irfft(f_new_hat * f_new_hat, n=n_fft)[:2*N-1]
                val_new = np.max(conv_new) / (N * N)
                
                if val_new < best_val:
                    best_val = val_new
                    best_alpha = trial_alpha
                    best_candidate = h_new
            
            if best_candidate is not None and best_val < current_val:
                best_f = best_candidate
                alpha_small = min(alpha_small * 1.05, 0.02)
            else:
                alpha_small *= 0.8
                if alpha_small < 1e-7:
                    break
    
    # Ensure nonnegative
    best_f = np.maximum(best_f, 0.0)
    return [float(x) for x in best_f]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
