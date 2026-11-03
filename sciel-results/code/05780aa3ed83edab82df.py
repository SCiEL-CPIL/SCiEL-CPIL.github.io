"""
First autocorrelation inequality: minimize normalized max of autoconvolution
for a nonnegative step function with 600 equal intervals on [-1/4, 1/4].

Strategy:
1. Parametric initialization with cosine bumps (smooth family)
2. FFT-based local refinement using smooth peak masks and backtracking line search
3. Haar wavelet coordinate descent to exploit piecewise-constant structure
   and escape local minima of time-domain methods
4. Final high-precision polish

The key insight is to combine the smooth sensitivity-based gradient approach
(FFT-accelerated) with a greedy coordinate descent in the Haar wavelet basis.
The Haar basis is natural for piecewise-constant functions and the orthogonality
allows us to make independent adjustments that reduce peak autoconvolution
without immediately undoing previous improvements.
"""

import numpy as np
from scipy.optimize import minimize
import time

N = 600
TIME_LIMIT = 148  # safety margin before 150s kill

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
    # Use a numerically stable log-sum-exp
    m = np.max(beta * shifted)
    log_sum = m / beta + np.log(np.sum(np.exp(beta * shifted - m))) / beta
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
    """Local refinement using FFT-based sensitivity peak reduction with adaptive
    threshold, smooth peak mask, and backtracking line search."""
    best_h = h.copy()
    best_val = compute_score(h)
    current_h = h.copy()
    alpha = 0.03
    threshold_frac = 0.97
    n_fft = 2 * N
    
    no_improve_count = 0
    max_no_improve = 15
    
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
        
        threshold = max_c * threshold_frac
        lower_thresh = max_c * 0.95
        peak_mask = np.clip((conv - lower_thresh) / (threshold - lower_thresh + 1e-15), 0, 1)
        
        pm_hat = np.fft.rfft(peak_mask, n=n_fft)
        ch_hat = np.fft.rfft(current_h, n=n_fft)
        sens = np.fft.irfft(pm_hat * np.conj(ch_hat), n=n_fft)[:N]
        
        sens_mean = np.mean(sens)
        sens_std = np.std(sens) + 1e-12
        sens_norm = (sens - sens_mean) / sens_std
        
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
            
            f_new_hat = np.fft.rfft(h_new, n=n_fft)
            conv_new = np.fft.irfft(f_new_hat * f_new_hat, n=n_fft)[:2*N-1]
            val_new = np.max(conv_new) / (N * N)
            
            if val_new < best_val_candidate:
                best_val_candidate = val_new
                best_alpha = trial_alpha
                best_h_candidate = h_new
        
        if best_h_candidate is not None and best_val_candidate < current_val:
            current_h = best_h_candidate
            if best_alpha > alpha:
                alpha = min(alpha * 1.1, 0.2)
            else:
                alpha = max(alpha * 0.9, 0.01)
        else:
            alpha *= 0.7
            if alpha < 1e-6:
                break
    
    return best_h, best_val

def haar_wavelet_refinement(h, start_time, max_time):
    """Greedy coordinate descent in Haar wavelet basis to exploit piecewise-constant
    structure and escape local minima of time-domain gradient methods.
    
    Uses proper Haar wavelet structure: for each level k and block b,
    the wavelet adds +delta/sqrt(2) to first half and -delta/sqrt(2) to second half.
    """
    best_h = h.copy()
    best_val = compute_score(h)
    current_h = h.copy()
    n_fft = 2 * N
    
    # Precompute Haar wavelet basis vectors for each (level, block) pair
    # A Haar wavelet at level k (0-indexed) has support of size 2^(k+1)
    # and is +1/sqrt(2) on first half, -1/sqrt(2) on second half
    haar_bases = []  # list of (indices_pos, indices_neg, scale)
    n_levels = int(np.log2(N))
    for level in range(n_levels + 1):
        block_size = 2 ** level
        n_blocks = N // (2 * block_size)
        for b in range(n_blocks):
            start = b * 2 * block_size
            pos_idx = np.arange(start, start + block_size)
            neg_idx = np.arange(start + block_size, start + 2 * block_size)
            scale = 1.0 / np.sqrt(2.0)
            haar_bases.append((pos_idx, neg_idx, scale))
    
    alpha = 0.05
    no_improve = 0
    max_no_improve = 40
    
    # Shuffle the order of Haar bases to avoid systematic bias
    rng = np.random.RandomState(42)
    
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
            no_improve = 0
        else:
            no_improve += 1
        
        if no_improve > max_no_improve:
            break
        
        # Try multiple random Haar basis updates
        best_candidate = None
        best_val_candidate = current_val
        best_delta = 0.0
        
        n_trials = min(20, len(haar_bases))
        trial_indices = rng.choice(len(haar_bases), size=n_trials, replace=False)
        
        for idx in trial_indices:
            pos_idx, neg_idx, scale = haar_bases[idx]
            
            for delta in [alpha * 0.5, alpha, alpha * 2.0, -alpha * 0.5, -alpha, -alpha * 2.0]:
                h_new = current_h.copy()
                h_new[pos_idx] += delta * scale
                h_new[neg_idx] -= delta * scale
                h_new = np.maximum(h_new, 0.0)
                
                s_new = np.sum(h_new)
                if s_new < 1e-12:
                    continue
                h_new = h_new * (N / s_new)
                
                conv_new = fft_convolve(h_new)
                val_new = np.max(conv_new) / (N * N)
                
                if val_new < best_val_candidate:
                    best_val_candidate = val_new
                    best_delta = delta
                    best_candidate = h_new
        
        if best_candidate is not None and best_val_candidate < current_val:
            current_h = best_candidate
            alpha = min(alpha * 1.05, 0.3)
        else:
            alpha *= 0.85
            if alpha < 1e-7:
                break
    
    return best_h, best_val

def spectral_refinement(h, start_time, max_time):
    """Spectral coordinate descent: optimize magnitude/phase per frequency bin
    with conjugate symmetry enforced. This decouples frequency bins and provides
    a different optimization path than time-domain methods."""
    best_h = h.copy()
    best_val = compute_score(h)
    current_h = h.copy()
    n_fft = 2 * N
    
    alpha = 0.02
    no_improve = 0
    max_no_improve = 25
    
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
            no_improve = 0
        else:
            no_improve += 1
        
        if no_improve > max_no_improve:
            break
        
        # Compute sensitivity in frequency domain
        # The autoconvolution in frequency domain is |F|^2
        # We want to reduce the peak of the time-domain convolution
        # Use peak mask to weight frequency bins by their contribution to peaks
        
        threshold = max_c * 0.98
        lower_thresh = max_c * 0.95
        peak_mask = np.clip((conv - lower_thresh) / (threshold - lower_thresh + 1e-15), 0, 1)
        
        pm_hat = np.fft.rfft(peak_mask, n=n_fft)
        ch_hat = np.fft.rfft(current_h, n=n_fft)
        
        # Sensitivity: how much does each frequency bin contribute to peak convolution?
        # d(conv)/d(f_hat[k]) = 2 * f_hat[k] * pm_hat[-k] (with conjugate symmetry)
        # In practice, we compute the gradient of the smooth max w.r.t. f_hat
        
        sens_freq = np.fft.irfft(pm_hat * np.conj(ch_hat), n=n_fft)[:N]
        
        # Also compute a frequency-domain sensitivity: which bins contribute most to peaks?
        # The contribution of bin k to the convolution at lag l is related to
        # |f_hat[k]|^2 * exp(2*pi*i*k*l/n)
        # For peak reduction, we want to reduce bins that contribute to peak lags
        
        # Compute per-bin importance based on peak mask
        bin_importance = np.abs(pm_hat * np.conj(ch_hat))
        bin_importance_norm = bin_importance / (np.sum(bin_importance) + 1e-15)
        
        # Try adjusting individual frequency bins
        best_candidate = None
        best_val_candidate = current_val
        best_bin = -1
        best_delta = 0.0
        
        # Sample a few frequency bins to try
        n_bins = len(f_hat)
        trial_bins = np.random.choice(n_bins, size=min(10, n_bins), replace=False)
        
        for k in trial_bins:
            for delta in [alpha * 0.5, alpha, alpha * 2.0, -alpha * 0.5, -alpha]:
                h_new = current_h.copy()
                f_new_hat = f_hat.copy()
                
                # Adjust magnitude of bin k (and its conjugate)
                if k == 0 or k == n_fft // 2:
                    # DC or Nyquist: real-valued
                    f_new_hat[k] += delta
                else:
                    # Adjust magnitude while preserving phase
                    mag = np.abs(f_new_hat[k])
                    phase = np.angle(f_new_hat[k])
                    new_mag = max(mag + delta, 0.0)
                    f_new_hat[k] = new_mag * np.exp(1j * phase)
                
                h_new = np.fft.irfft(f_new_hat, n=n_fft)[:N]
                h_new = np.maximum(h_new, 0.0)
                
                s_new = np.sum(h_new)
                if s_new < 1e-12:
                    continue
                h_new = h_new * (N / s_new)
                
                f_new_hat2 = np.fft.rfft(h_new, n=n_fft)
                conv_new = np.fft.irfft(f_new_hat2 * f_new_hat2, n=n_fft)[:2*N-1]
                val_new = np.max(conv_new) / (N * N)
                
                if val_new < best_val_candidate:
                    best_val_candidate = val_new
                    best_bin = k
                    best_delta = delta
                    best_candidate = h_new
        
        if best_candidate is not None and best_val_candidate < current_val:
            current_h = best_candidate
            alpha = min(alpha * 1.05, 0.15)
        else:
            alpha *= 0.85
            if alpha < 1e-7:
                break
    
    return best_h, best_val


def construct():
    start_time = time.time()
    
    # Initial best: constant function
    best_f = np.ones(N)
    best_score = compute_score(best_f)
    
    # Phase 1: Parametric initialization with cosine bumps
    # Reserve time: 20% for parametric, 25% for local refinement, 20% for spectral,
    # 20% for wavelet, 15% for polish
    param_budget_end = start_time + (TIME_LIMIT - 5) * 0.20
    
    for n_c in [2, 3, 4]:
        if time.time() > param_budget_end:
            break
        
        rng = np.random.RandomState(42 + n_c)
        best_params = None
        best_obj = np.inf
        
        betas = [10.0, 20.0, 30.0]
        
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
    
    # Phase 2: FFT-based local refinement
    if time.time() < TIME_LIMIT - 60:
        refined_f, refined_score = local_refinement(
            best_f, start_time, TIME_LIMIT - 60
        )
        if refined_score < best_score:
            best_f = refined_f
            best_score = refined_score
    
    # Phase 3: Spectral coordinate descent
    if time.time() < TIME_LIMIT - 35:
        spectral_f, spectral_score = spectral_refinement(
            best_f, start_time, TIME_LIMIT - 35
        )
        if spectral_score < best_score:
            best_f = spectral_f
            best_score = spectral_score
    
    # Phase 4: Haar wavelet structured coordinate descent
    if time.time() < TIME_LIMIT - 15:
        wavelet_f, wavelet_score = haar_wavelet_refinement(
            best_f, start_time, TIME_LIMIT - 15
        )
        if wavelet_score < best_score:
            best_f = wavelet_f
            best_score = wavelet_score
    
    # Phase 5: Final high-precision polish with adaptive beta
    if time.time() < TIME_LIMIT - 2:
        best_f = best_f.copy()
        alpha_small = 0.005
        n_fft = 2 * N
        beta_adaptive = 50.0
        
        for _ in range(300):
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
            
            # Adaptive peak mask: tighter as we converge
            threshold = max_c * 0.999
            lower_thresh = max_c * 0.997
            peak_mask = np.clip((conv - lower_thresh) / (threshold - lower_thresh + 1e-15), 0, 1)
            
            pm_hat = np.fft.rfft(peak_mask, n=n_fft)
            ch_hat = np.fft.rfft(best_f, n=n_fft)
            sens = np.fft.irfft(pm_hat * np.conj(ch_hat), n=n_fft)[:N]
            
            sens_mean = np.mean(sens)
            sens_std = np.std(sens) + 1e-12
            sens_norm = (sens - sens_mean) / sens_std
            
            best_alpha = alpha_small
            best_candidate = None
            best_val = current_val
            
            for trial_alpha in [alpha_small, alpha_small * 0.5, alpha_small * 2.0, alpha_small * 0.25]:
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
                alpha_small = min(alpha_small * 1.03, 0.015)
                # Increase beta for sharper peak focus as we converge
                beta_adaptive = min(beta_adaptive * 1.01, 200.0)
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
