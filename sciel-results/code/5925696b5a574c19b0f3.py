"""
First autocorrelation inequality: minimize normalized max of autoconvolution
for a nonnegative step function with 600 equal intervals on [-1/4, 1/4].

Strategy:
1. Parametric initialization with cosine bumps
2. Spectral magnitude optimization
3. Block-level simulated annealing with peak-directed perturbations
4. Final local polish with sensitivity-based refinement
5. Additional: smooth surrogate optimization using log-sum-exp for the max
   and epigraph-style auxiliary variable to better handle the nonsmooth peak
"""

import numpy as np
from scipy.optimize import minimize, minimize_scalar
import time

N = 600
TIME_LIMIT = 146  # safety margin before 150s kill

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

def fft_convolve(f):
    """Fast autoconvolution via FFT."""
    n_fft = 2 * N
    f_hat = np.fft.rfft(f, n=n_fft)
    conv = np.fft.irfft(f_hat * f_hat, n=n_fft)[:2*N-1]
    return conv

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

def spectral_refinement(h, start_time, max_time):
    """Spectral magnitude optimization: fit a parametric model to |F(k)|,
    optimize parameters to minimize peak autoconvolution, then project back
    to nonnegative time domain."""
    best_h = h.copy()
    best_val = compute_score(h)
    n_fft = 2 * N
    
    # Get current spectrum
    f_hat = np.fft.rfft(h, n=n_fft)
    mag = np.abs(f_hat)
    n_freqs = len(mag)
    
    # Parametric model: log-magnitude as sum of Lorentzian bumps in frequency
    def spectral_objective(params, n_bumps):
        centers = params[:n_bumps]
        widths = params[n_bumps:2*n_bumps]
        heights = params[2*n_bumps:3*n_bumps]
        
        widths = np.abs(widths) + 0.01
        heights = np.abs(heights)
        
        freqs = np.arange(n_freqs)
        log_mag = np.zeros(n_freqs)
        for c, w, hgt in zip(centers, widths, heights):
            log_mag += hgt / (1.0 + ((freqs - c) / w) ** 2)
        
        mag_model = np.exp(log_mag)
        
        # Reconstruct time domain with conjugate symmetry
        f_recon = np.zeros(n_fft)
        f_recon[:n_freqs] = mag_model
        f_recon[n_freqs-1:0:-1] = mag_model[n_freqs-1:0:-1]
        
        f_time = np.fft.irfft(f_recon, n=n_fft)[:N]
        f_time = np.maximum(f_time, 0.0)
        
        s = np.sum(f_time)
        if s < 1e-12:
            return 1e10
        f_time = f_time * (N / s)
        
        conv = fft_convolve(f_time)
        return np.max(conv) / (N * N)
    
    # Fit initial parameters from current spectrum
    rng = np.random.RandomState(42)
    best_params = None
    best_obj = np.inf
    
    for n_bumps in [3, 5, 8]:
        if time.time() > max_time:
            break
        
        for restart in range(3):
            if time.time() > max_time:
                break
            
            # Initialize from peaks in current magnitude spectrum
            mag_smooth = mag.copy()
            # Find local maxima in magnitude
            peaks = []
            for i in range(1, n_freqs-1):
                if mag[i] > mag[i-1] and mag[i] > mag[i+1] and mag[i] > np.mean(mag) * 0.5:
                    peaks.append(i)
            
            if len(peaks) >= n_bumps:
                # Select top n_bumps peaks
                peak_vals = [mag[p] for p in peaks]
                top_idx = np.argsort(peak_vals)[-n_bumps:]
                centers = np.array([peaks[i] for i in top_idx], dtype=float)
            else:
                centers = rng.uniform(0, n_freqs-1, size=n_bumps)
            
            widths = rng.uniform(2.0, 20.0, size=n_bumps)
            # Estimate heights from log-magnitude at peaks
            heights = np.zeros(n_bumps)
            for i, c in enumerate(centers):
                c_int = int(np.clip(c, 0, n_freqs-1))
                heights[i] = np.log(mag[c_int] + 1e-10)
            
            x0 = np.concatenate([centers, widths, heights])
            
            try:
                result = minimize(
                    spectral_objective, x0, args=(n_bumps,),
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 0.1, 'fatol': 1e-5}
                )
                if result.fun < best_obj:
                    best_obj = result.fun
                    best_params = result.x
            except Exception:
                pass
    
    if best_params is not None:
        # Reconstruct from optimized spectrum
        n_bumps = best_params.size // 3
        centers = best_params[:n_bumps]
        widths = np.abs(best_params[n_bumps:2*n_bumps]) + 0.01
        heights = np.abs(best_params[2*n_bumps:3*n_bumps])
        
        freqs = np.arange(n_freqs)
        log_mag = np.zeros(n_freqs)
        for c, w, hgt in zip(centers, widths, heights):
            log_mag += hgt / (1.0 + ((freqs - c) / w) ** 2)
        
        mag_model = np.exp(log_mag)
        
        f_recon = np.zeros(n_fft)
        f_recon[:n_freqs] = mag_model
        f_recon[n_freqs-1:0:-1] = mag_model[n_freqs-1:0:-1]
        
        f_time = np.fft.irfft(f_recon, n=n_fft)[:N]
        f_time = np.maximum(f_time, 0.0)
        
        s = np.sum(f_time)
        if s > 1e-12:
            f_time = f_time * (N / s)
            val = compute_score(f_time)
            if val < best_val:
                best_h = f_time
                best_val = val
    
    return best_h, best_val

def block_simulated_annealing(h, start_time, max_time):
    """Block-level simulated annealing with peak-directed perturbations."""
    best_h = h.copy()
    best_val = compute_score(h)
    current_h = h.copy()
    n_fft = 2 * N
    
    # Precompute current FFT for faster evaluation
    current_fft = np.fft.rfft(current_h, n=n_fft)
    current_conv = np.fft.irfft(current_fft * current_fft, n=n_fft)[:2*N-1]
    current_val = np.max(current_conv) / (N * N)
    
    # Cooling schedule
    T_start = 0.15
    T_end = 0.0005
    T_current = T_start
    
    # Block sizes to try
    block_sizes = [4, 8, 16, 32, 64, 128]
    
    no_improve = 0
    max_no_improve = 120
    total_steps = 0
    
    while time.time() - start_time < max_time:
        total_steps += 1
        s = np.sum(current_h)
        if s < 1e-12:
            break
        current_h = current_h * (N / s)
        
        # Determine peak location for directed perturbations
        peak_idx = np.argmax(current_conv)
        
        # Pick a random block, biased toward peak region
        bs = block_sizes[np.random.randint(len(block_sizes))]
        n_blocks = N // bs
        if n_blocks < 2:
            bs = 4
            n_blocks = N // bs
        
        # 30% chance to pick a block near the peak
        if np.random.random() < 0.3:
            # Map conv peak index to block index
            # conv has length 2*N-1, peak_idx maps to time offset
            time_offset = peak_idx - (N - 1)
            center_block = max(0, min(n_blocks - 1, (time_offset + N//2) // bs))
            block_idx = max(0, min(n_blocks - 1, center_block + np.random.randint(-2, 3)))
        else:
            block_idx = np.random.randint(0, n_blocks)
        
        start_idx = block_idx * bs
        end_idx = start_idx + bs
        
        # Generate perturbation
        mode = np.random.randint(5)
        
        h_new = current_h.copy()
        if mode == 0:
            # Scale block
            scale = np.random.uniform(0.5, 1.5)
            h_new[start_idx:end_idx] *= scale
        elif mode == 1:
            # Shift mass within block
            half = bs // 2
            delta = np.random.uniform(-0.3, 0.3) * np.mean(current_h[start_idx:end_idx])
            h_new[start_idx:start_idx+half] -= delta
            h_new[start_idx+half:end_idx] += delta
        elif mode == 2:
            # Add/remove Gaussian bump
            center = start_idx + bs // 2
            sigma = max(2, bs // 4)
            x = np.arange(start_idx, end_idx)
            bump = np.exp(-((x - center) ** 2) / (2 * sigma ** 2))
            amp = np.random.uniform(-0.25, 0.25) * np.mean(current_h)
            h_new[start_idx:end_idx] += amp * bump
        elif mode == 3:
            # Smooth the block with neighbors (low-pass)
            if start_idx >= 2 and end_idx <= N - 2:
                left = np.mean(current_h[start_idx-2:start_idx])
                right = np.mean(current_h[end_idx:end_idx+2])
                h_new[start_idx:end_idx] = 0.7 * current_h[start_idx:end_idx] + 0.15 * left + 0.15 * right
            else:
                h_new[start_idx:end_idx] *= np.random.uniform(0.8, 1.2)
        else:
            # Random reweight: multiply by a smooth random factor
            x = np.linspace(0, 1, bs)
            factor = 1.0 + np.random.uniform(-0.3, 0.3) * np.cos(np.pi * x)
            h_new[start_idx:end_idx] *= factor
        
        h_new = np.maximum(h_new, 0.0)
        s_new = np.sum(h_new)
        if s_new < 1e-12:
            continue
        h_new = h_new * (N / s_new)
        
        # Fast evaluation using FFT
        new_fft = np.fft.rfft(h_new, n=n_fft)
        new_conv = np.fft.irfft(new_fft * new_fft, n=n_fft)[:2*N-1]
        val_new = np.max(new_conv) / (N * N)
        
        # Accept if better, or with probability exp(-(val_new - val_cur)/T)
        if val_new < current_val or np.random.random() < np.exp(-(val_new - current_val) / max(T_current, 1e-10)):
            current_h = h_new
            current_fft = new_fft
            current_conv = new_conv
            current_val = val_new
            no_improve = 0
        else:
            no_improve += 1
        
        if val_new < best_val:
            best_val = val_new
            best_h = h_new.copy()
            no_improve = 0
        
        # Cool down with geometric schedule
        cooling_rate = (T_end / T_start) ** (1.0 / max(800, total_steps))
        T_current = max(T_current * cooling_rate, T_end)
        
        if no_improve > max_no_improve:
            # Reset to best and restart with higher temperature
            current_h = best_h.copy()
            current_fft = np.fft.rfft(current_h, n=n_fft)
            current_conv = np.fft.irfft(current_fft * current_fft, n=n_fft)[:2*N-1]
            current_val = np.max(current_conv) / (N * N)
            T_current = T_start * 0.6
            no_improve = 0
    
    return best_h, best_val

def local_polish(h, start_time, max_time):
    """Final local polish with iterative peak-directed sensitivity refinement."""
    best_h = h.copy()
    best_val = compute_score(h)
    current_h = h.copy()
    alpha = 0.03
    n_fft = 2 * N
    
    no_improve_count = 0
    max_no_improve = 80
    
    # Track peak location history for adaptive mask width
    peak_width = 0.15  # fraction of conv length in the peak region
    
    while time.time() - start_time < max_time:
        s = np.sum(current_h)
        if s < 1e-12:
            break
        current_h = current_h * (N / s)
        
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
        
        # Adaptive peak mask: find the region where conv is close to max
        # Use a Gaussian-like falloff centered at the peak
        peak_idx = np.argmax(conv)
        conv_len = len(conv)
        
        # Determine mask width based on how concentrated the peak is
        # Find where conv drops below 80% of max
        threshold_80 = max_c * 0.80
        above_80 = conv >= threshold_80
        if np.any(above_80):
            idxs = np.where(above_80)[0]
            mask_width = max(5, (idxs[-1] - idxs[0] + 1))
        else:
            mask_width = conv_len // 10
        
        # Build smooth peak mask
        center = peak_idx
        half_w = max(2, mask_width // 2)
        x = np.arange(conv_len)
        # Smooth Gaussian-like mask
        sigma = half_w / 2.0
        peak_mask = np.exp(-0.5 * ((x - center) / sigma) ** 2)
        
        # Also add a secondary mask for near-peak regions to avoid over-concentrating
        sigma2 = half_w * 2.0
        secondary_mask = 0.3 * np.exp(-0.5 * ((x - center) / sigma2) ** 2)
        peak_mask = peak_mask + secondary_mask
        peak_mask = peak_mask / peak_mask.max()
        
        pm_hat = np.fft.rfft(peak_mask, n=n_fft)
        ch_hat = np.fft.rfft(current_h, n=n_fft)
        sens = np.fft.irfft(pm_hat * np.conj(ch_hat), n=n_fft)[:N]
        
        sens_mean = np.mean(sens)
        sens_std = np.std(sens) + 1e-12
        sens_norm = (sens - sens_mean) / sens_std
        
        best_alpha = alpha
        best_h_candidate = None
        best_val_candidate = current_val
        
        # Try multiple alpha values
        for trial_alpha in [alpha * 0.25, alpha * 0.5, alpha, alpha * 1.5, alpha * 2.0, alpha * 3.0]:
            h_new = current_h * (1.0 - trial_alpha * sens_norm)
            h_new = np.maximum(h_new, 0.0)
            
            s_new = np.sum(h_new)
            if s_new < 1e-12:
                continue
            h_new = h_new * (N / s_new)
            
            f_new_hat = np.fft.rfft(h_new, n=n_fft)
            conv_new = np.fft.irfft(f_new_hat * f_new_hat, n=n_fft)[:2*N-1]
            val_new = np.max(conv_new) / (N * N)
            
            if val_new < best_val_candidate:
                best_val_candidate = val_new
                best_alpha = trial_alpha
                best_h_candidate = h_new
        
        if best_h_candidate is not None and best_val_candidate < current_val:
            current_h = best_h_candidate
            alpha = min(alpha * 1.1, 0.20)
            # Narrow the peak width slightly as we converge
            peak_width = max(0.05, peak_width * 0.98)
        else:
            alpha *= 0.75
            # Widen the mask when stuck
            peak_width = min(0.30, peak_width * 1.1)
            if alpha < 1e-7:
                break
    
    return best_h, best_val

def smooth_surrogate_polish(h, start_time, max_time):
    """Use a smooth surrogate (log-sum-exp) for the max in the autoconvolution,
    then do a final projected gradient descent on the time-domain signal
    using the gradient of the smooth objective. This handles the nonsmooth
    max more gracefully and can find better local minima."""
    best_h = h.copy()
    best_val = compute_score(h)
    current_h = h.copy()
    n_fft = 2 * N
    
    # Start with a moderate beta and increase it gradually
    beta_start = 5.0
    beta_end = 50.0
    lr = 0.01
    
    no_improve = 0
    max_no_improve = 60
    total_iters = 0
    
    while time.time() - start_time < max_time:
        total_iters += 1
        
        s = np.sum(current_h)
        if s < 1e-12:
            break
        current_h = current_h * (N / s)
        
        # Compute smooth surrogate objective and its gradient
        f_hat = np.fft.rfft(current_h, n=n_fft)
        conv = np.fft.irfft(f_hat * f_hat, n=n_fft)[:2*N-1]
        
        # Smooth max via log-sum-exp
        conv_max = np.max(conv)
        if conv_max < 1e-15:
            break
        conv_norm = conv / conv_max
        beta = beta_start + (beta_end - beta_start) * min(1.0, total_iters / 200.0)
        shifted = conv_norm - np.max(conv_norm)
        log_sum = np.log(np.sum(np.exp(beta * shifted))) / beta
        smooth_obj = conv_max * (1.0 + log_sum) / (N * N)
        
        # Gradient of smooth max w.r.t. conv: softmax weights
        weights = np.exp(beta * (conv_norm - np.max(conv_norm)))
        weights /= np.sum(weights)
        
        # Gradient of conv w.r.t. h: conv is autocorrelation, so
        # d(conv[k])/d(h[i]) = h[k-i] for valid indices (with appropriate shifts)
        # For the full convolution, grad_h[i] = sum_k weights[k] * d(conv[k])/d(h[i])
        # This is equivalent to: grad_h = convolve(weights, h) appropriately
        # More precisely, conv = h * h (full), so d(conv)/dh is like a cross-correlation
        # For simplicity, use the FFT-based approach:
        # The gradient of ||conv||^2 w.r.t. h is 4 * ifft(fft(h) * fft(conv_weights))
        # But we want gradient of smooth_max(conv), which is weighted sum of conv[k]
        # d(smooth_max)/dh[i] = sum_k weights[k] * d(conv[k])/dh[i]
        # conv[k] = sum_j h[j] * h[k-j]
        # d(conv[k])/dh[i] = h[k-i] + h[i-k] (for the full convolution)
        # So d(smooth_max)/dh[i] = sum_k weights[k] * (h[k-i] + h[i-k])
        # This is 2 * (weights * h) cross-correlated, which in FFT is:
        w_hat = np.fft.rfft(weights, n=n_fft)
        h_hat = np.fft.rfft(current_h, n=n_fft)
        grad_conv = np.fft.irfft(w_hat * h_hat, n=n_fft)[:N]
        # The full gradient includes both h[k-i] and h[i-k], so:
        grad_full = grad_conv + np.fft.irfft(np.conj(w_hat) * h_hat, n=n_fft)[:N]
        grad_full = (grad_full + grad_full[::-1]) / 2.0  # ensure symmetry
        
        # Actually, let's use a simpler approach: finite difference or direct computation
        # For the autocorrelation conv = ifft(fft(h)^2), the gradient w.r.t. h is:
        # grad = ifft(fft(weights) * fft(h)) * 2 (accounting for the square)
        # Let's use the direct formula:
        # d/dh_i [sum_k w_k * conv_k] where conv_k = sum_j h_j h_{k-j}
        # = sum_k w_k * (h_{k-i} + h_{i-k})
        # = 2 * sum_k w_k * h_{k-i} (by symmetry of autocorrelation)
        # In frequency domain: ifft(fft(w) * fft(h)) gives sum_k w_k h_{k-i}
        # So grad = 2 * ifft(fft(w) * fft(h))
        grad = 2.0 * np.fft.irfft(w_hat * h_hat, n=n_fft)[:N]
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-15:
            no_improve += 1
        else:
            grad /= grad_norm
        
        # Projected gradient step
        h_new = current_h - lr * grad
        h_new = np.maximum(h_new, 0.0)
        s_new = np.sum(h_new)
        if s_new < 1e-12:
            no_improve += 1
        else:
            h_new = h_new * (N / s_new)
            val_new = compute_score(h_new)
            
            if val_new < best_val:
                best_val = val_new
                best_h = h_new.copy()
                no_improve = 0
                lr = min(lr * 1.05, 0.05)
            elif val_new < best_val + 0.001:
                # Accept slightly worse to escape local minima
                current_h = h_new
                no_improve += 1
            else:
                no_improve += 1
                lr *= 0.8
        
        current_h = h_new if no_improve <= max_no_improve else best_h.copy()
        
        if no_improve > max_no_improve:
            break
    
    return best_h, best_val

def construct():
    start_time = time.time()
    
    # Initial best: constant function
    best_f = np.ones(N)
    best_score = compute_score(best_f)
    
    # Phase 1: Parametric initialization with cosine bumps
    param_budget_end = start_time + (TIME_LIMIT - 5) * 0.08
    
    for n_c in [2, 3, 4]:
        if time.time() > param_budget_end:
            break
        
        rng = np.random.RandomState(42 + n_c)
        best_params = None
        best_obj = np.inf
        
        for restart in range(3):
            if time.time() > param_budget_end:
                break
            
            centers = rng.uniform(-0.15, 0.15, size=n_c)
            widths = rng.uniform(0.06, 0.30, size=n_c)
            heights = rng.uniform(0.3, 2.5, size=n_c)
            
            x0 = np.concatenate([centers, widths, heights])
            
            try:
                result = minimize(
                    objective, x0, args=(n_c, 20.0),
                    method='Nelder-Mead',
                    options={'maxiter': 400, 'xatol': 1e-5, 'fatol': 1e-7}
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
    
    # Phase 2: Spectral magnitude optimization
    if time.time() < TIME_LIMIT - 70:
        spectral_f, spectral_score = spectral_refinement(
            best_f, start_time, TIME_LIMIT - 70
        )
        if spectral_score < best_score:
            best_f = spectral_f
            best_score = spectral_score
    
    # Phase 3: Block-level simulated annealing
    if time.time() < TIME_LIMIT - 40:
        sa_f, sa_score = block_simulated_annealing(
            best_f, start_time, TIME_LIMIT - 40
        )
        if sa_score < best_score:
            best_f = sa_f
            best_score = sa_score
    
    # Phase 4: Smooth surrogate polish (new)
    if time.time() < TIME_LIMIT - 15:
        smooth_f, smooth_score = smooth_surrogate_polish(
            best_f, start_time, TIME_LIMIT - 15
        )
        if smooth_score < best_score:
            best_f = smooth_f
            best_score = smooth_score
    
    # Phase 5: Final local polish
    if time.time() < TIME_LIMIT - 1:
        polished_f, polished_score = local_polish(
            best_f, start_time, TIME_LIMIT - 1
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
