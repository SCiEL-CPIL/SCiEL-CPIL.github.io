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

def local_refinement(h, start_time, max_time):
    """Local refinement using FFT-based sensitivity peak reduction with
    adaptive threshold and coordinate descent polish.
    """
    best_h = h.copy()
    best_val = compute_score(h)
    current_h = h.copy()
    alpha = 0.03
    threshold_frac = 0.97
    n_fft = 2 * N  # size for full convolution via FFT
    stall_count = 0
    max_stall = 20
    
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
            stall_count = 0
        else:
            stall_count += 1
            if stall_count > max_stall:
                # Try coordinate descent polish
                current_h = _coordinate_descent_polish(current_h, start_time, max_time)
                s = np.sum(current_h)
                if s < 1e-12:
                    break
                current_h = current_h * (N / s)
                f_hat = np.fft.rfft(current_h, n=n_fft)
                conv = np.fft.irfft(f_hat * f_hat, n=n_fft)[:2*N-1]
                current_val = np.max(conv) / (N * N)
                if current_val < best_val:
                    best_val = current_val
                    best_h = current_h.copy()
                    stall_count = 0
                else:
                    break
        
        # Adaptive threshold: tighten as we converge
        threshold_frac = max(0.92, 0.97 - 0.015 * stall_count)
        
        # Weighted peak mask: emphasize the actual maximum more
        threshold = max_c * threshold_frac
        # Use soft weighting: peaks closer to max get higher weight
        peak_weights = np.maximum(0.0, (conv - threshold) / (max_c - threshold + 1e-15))
        peak_weights += 1.0  # base weight of 1 for all above threshold
        peak_mask = peak_weights * (conv >= threshold)
        
        # Compute sensitivity via FFT
        pm_hat = np.fft.rfft(peak_mask, n=n_fft)
        ch_hat = np.fft.rfft(current_h, n=n_fft)
        sens = np.fft.irfft(pm_hat * np.conj(ch_hat), n=n_fft)[:N]
        
        # Normalize sensitivity
        sens_mean = np.mean(sens)
        sens_std = np.std(sens) + 1e-12
        sens_norm = (sens - sens_mean) / sens_std
        
        # Reduce bins with high sensitivity (they contribute to peaks)
        # Use a smoother update to avoid oscillation
        h_new = current_h * (1.0 - alpha * np.clip(sens_norm, -2.0, 2.0))
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
            alpha = min(alpha * 1.03, 0.2)
        else:
            alpha *= 0.8
            if alpha < 1e-9:
                break
    
    return best_h, best_val

def _coordinate_descent_polish(h, start_time, max_time):
    """Coordinate descent: for each bin, find optimal value that minimizes
    the peak autoconvolution contribution using efficient updates.
    """
    n_fft = 2 * N
    h = h.copy()
    s = np.sum(h)
    if s < 1e-12:
        return h
    h = h * (N / s)
    
    # Precompute current FFT and convolution
    f_hat = np.fft.rfft(h, n=n_fft)
    conv = np.fft.irfft(f_hat * f_hat, n=n_fft)[:2*N-1]
    current_val = np.max(conv) / (N * N)
    
    # Do a few passes of coordinate updates with limited binary search
    for _pass in range(2):
        if time.time() - start_time > max_time - 5:
            break
        
        # Process bins in random order to avoid bias
        order = np.random.RandomState(42 + _pass).permutation(N)
        
        for idx in order:
            if time.time() - start_time > max_time - 5:
                break
            
            i = int(idx)
            old_val = h[i]
            if old_val < 1e-15:
                continue
            
            # Try a few candidate values: 0, 0.5*old, 1.5*old, 2*old
            candidates = [0.0, old_val * 0.5, old_val * 1.5, old_val * 2.0]
            best_candidate_val = old_val
            best_candidate_score = current_val
            
            for cand in candidates:
                if cand < 0:
                    continue
                h[i] = cand
                s_temp = np.sum(h)
                if s_temp < 1e-12:
                    h[i] = old_val
                    continue
                h_temp = h * (N / s_temp)
                f_temp_hat = np.fft.rfft(h_temp, n=n_fft)
                conv_temp = np.fft.irfft(f_temp_hat * f_temp_hat, n=n_fft)[:2*N-1]
                val_temp = np.max(conv_temp) / (N * N)
                
                if val_temp < best_candidate_score:
                    best_candidate_score = val_temp
                    best_candidate_val = cand
            
            h[i] = best_candidate_val
            s = np.sum(h)
            if s > 1e-12:
                h = h * (N / s)
                f_hat = np.fft.rfft(h, n=n_fft)
                conv = np.fft.irfft(f_hat * f_hat, n=n_fft)[:2*N-1]
                current_val = np.max(conv) / (N * N)
    
    return h

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
        
        # Second refinement pass with slightly different initialization
        if time.time() < TIME_LIMIT - 3:
            # Perturb slightly to escape local minima
            rng_polish = np.random.RandomState(123)
            perturbed = best_f * (1.0 + 0.01 * rng_polish.randn(N))
            perturbed = np.maximum(perturbed, 0.0)
            s_pert = np.sum(perturbed)
            if s_pert > 1e-12:
                perturbed = perturbed * (N / s_pert)
                refined2_f, refined2_score = local_refinement(
                    perturbed, start_time, TIME_LIMIT - 3
                )
                if refined2_score < best_score:
                    best_f = refined2_f
                    best_score = refined2_score
    
    # Final high-precision polish: tighter threshold, smaller steps, adaptive
    if time.time() < TIME_LIMIT - 2:
        best_f = best_f.copy()
        alpha_small = 0.005
        threshold_final = 0.995
        n_fft = 2 * N
        best_val_polish = compute_score(best_f)
        
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
            
            if current_val < best_val_polish:
                best_val_polish = current_val
            
            # Weighted peak mask for finer control
            threshold = max_c * threshold_final
            peak_weights = np.maximum(0.0, (conv - threshold) / (max_c - threshold + 1e-15))
            peak_mask = peak_weights * (conv >= threshold) + (conv >= threshold)
            
            pm_hat = np.fft.rfft(peak_mask, n=n_fft)
            ch_hat = np.fft.rfft(best_f, n=n_fft)
            sens = np.fft.irfft(pm_hat * np.conj(ch_hat), n=n_fft)[:N]
            
            sens_mean = np.mean(sens)
            sens_std = np.std(sens) + 1e-12
            sens_norm = (sens - sens_mean) / sens_std
            
            # Smoother update with clipping
            h_new = best_f * (1.0 - alpha_small * np.clip(sens_norm, -3.0, 3.0))
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
                alpha_small = min(alpha_small * 1.01, 0.05)
            else:
                alpha_small *= 0.85
                if alpha_small < 1e-8:
                    break
    
    # Ensure nonnegative
    best_f = np.maximum(best_f, 0.0)
    return [float(x) for x in best_f]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
