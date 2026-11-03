"""
First autocorrelation inequality: minimize normalized max of autoconvolution
for a nonnegative step function with 600 equal intervals on [-1/4, 1/4].

Strategy:
1. Initialize with a smooth non-negative bump (raised cosine or Gaussian).
2. Perform spectral flattening: iteratively adjust frequency magnitudes to be
   more uniform, which tends to flatten the autocorrelation peak.
3. Perform time-domain peak shaving: use FFT-based sensitivity to reduce bins
   that contribute most to the current peak.
4. Use random restarts with different initial shapes to escape local minima.
"""

import numpy as np
import time

N = 600
TIME_LIMIT = 148  # Safety margin for 150s kill

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

def raised_cosine(width, N):
    """Generate a raised cosine bump of given width centered in the array."""
    x = np.linspace(-0.25, 0.25, N, endpoint=False)
    # Center at 0
    # Width is the full width of the bump
    mask = np.abs(x) <= width / 2.0
    f = np.zeros(N)
    f[mask] = np.cos(np.pi * (x[mask]) / width) ** 2
    return f

def gaussian(width, N):
    """Generate a Gaussian bump of given standard deviation."""
    x = np.linspace(-0.25, 0.25, N, endpoint=False)
    f = np.exp(-x**2 / (2 * width**2))
    return f

def spectral_flattening(f, start_time, max_time, beta=1.0):
    """
    Iteratively flatten the power spectrum to reduce autocorrelation peak.
    This works by adjusting the magnitude of each frequency bin to be closer
    to the mean magnitude, while maintaining non-negativity in time domain.
    """
    best_f = f.copy()
    best_val = compute_score(f)
    n_fft = 2 * N
    current_f = f.copy()
    
    # Normalize
    s = np.sum(current_f)
    if s < 1e-12: return best_f, best_val
    current_f = current_f * (N / s)
    
    no_improve = 0
    max_no_improve = 50
    alpha = 0.1
    
    while time.time() - start_time < max_time:
        f_hat = np.fft.rfft(current_f, n=n_fft)
        mags = np.abs(f_hat)
        phases = np.angle(f_hat)
        
        # Target: uniform magnitude
        mean_mag = np.mean(mags)
        
        # Adjust magnitudes towards the mean
        # new_mags = mags + alpha * (mean_mag - mags)
        # But we must be careful not to make them negative or zero
        new_mags = mags * (1.0 - alpha) + mean_mag * alpha
        new_mags = np.maximum(new_mags, 0.0)
        
        # Reconstruct
        new_f_hat = new_mags * np.exp(1j * phases)
        new_f = np.fft.irfft(new_f_hat, n=n_fft)[:N]
        
        # Enforce non-negativity
        new_f = np.maximum(new_f, 0.0)
        
        # Normalize
        s_new = np.sum(new_f)
        if s_new < 1e-12:
            break
        new_f = new_f * (N / s_new)
        
        # Evaluate
        val_new = compute_score(new_f)
        
        if val_new < best_val:
            best_val = val_new
            best_f = new_f.copy()
            no_improve = 0
            # Increase alpha slightly if improving
            alpha = min(alpha * 1.05, 0.5)
        else:
            no_improve += 1
            alpha *= 0.9
            if no_improve > max_no_improve:
                break
            if alpha < 1e-5:
                break
                
    return best_f, best_val

def peak_shaving(f, start_time, max_time, threshold_frac=0.95):
    """
    Refine by reducing bins that contribute to the peak of the autoconvolution.
    """
    best_f = f.copy()
    best_val = compute_score(f)
    n_fft = 2 * N
    current_f = f.copy()
    
    s = np.sum(current_f)
    if s < 1e-12: return best_f, best_val
    current_f = current_f * (N / s)
    
    no_improve = 0
    max_no_improve = 50
    alpha = 0.05
    
    while time.time() - start_time < max_time:
        f_hat = np.fft.rfft(current_f, n=n_fft)
        conv = np.fft.irfft(f_hat * f_hat, n=n_fft)[:2*N-1]
        max_c = np.max(conv)
        current_val = max_c / (N * N)
        
        if current_val < best_val:
            best_val = current_val
            best_f = current_f.copy()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve > max_no_improve:
                break
        
        # Find peak lags
        threshold = max_c * threshold_frac
        peak_mask = (conv >= threshold).astype(np.float64)
        
        # Sensitivity: cross-correlation of peak mask with current_f
        pm_hat = np.fft.rfft(peak_mask, n=n_fft)
        ch_hat = np.fft.rfft(current_f, n=n_fft)
        sens = np.fft.irfft(pm_hat * np.conj(ch_hat), n=n_fft)[:N]
        
        # Normalize sensitivity
        sens_mean = np.mean(sens)
        sens_std = np.std(sens) + 1e-12
        sens_norm = (sens - sens_mean) / sens_std
        
        # Reduce high sensitivity bins
        h_new = current_f * (1.0 - alpha * sens_norm)
        h_new = np.maximum(h_new, 0.0)
        
        s_new = np.sum(h_new)
        if s_new < 1e-12:
            break
        h_new = h_new * (N / s_new)
        
        val_new = compute_score(h_new)
        
        if val_new < current_val:
            current_f = h_new
            alpha = min(alpha * 1.02, 0.2)
        else:
            alpha *= 0.9
            if alpha < 1e-6:
                break

    return best_f, best_val

def construct():
    start_time = time.time()
    
    best_f = None
    best_score = np.inf
    
    # Try different initial shapes
    shapes = [
        raised_cosine(0.2, N),
        raised_cosine(0.1, N),
        raised_cosine(0.3, N),
        gaussian(0.05, N),
        gaussian(0.1, N),
        np.ones(N)
    ]
    
    # Reserve time for refinement
    refinement_budget = 60  # seconds
    
    for shape in shapes:
        if time.time() - start_time > 10:
            break
            
        # Normalize initial shape
        s = np.sum(shape)
        if s < 1e-12:
            continue
        f0 = shape * (N / s)
        
        # Quick spectral flattening
        f_flat, score_flat = spectral_flattening(
            f0, start_time, time.time() + 5
        )
        
        if score_flat < best_score:
            best_score = score_flat
            best_f = f_flat.copy()
            
        # If we have time, do peak shaving on the best candidate found so far
        # We will do a final global refinement on the best candidate at the end
    
    # Final refinement phase on the best candidate
    if best_f is None:
        # Fallback
        best_f = np.ones(N)
        best_f = best_f * (N / np.sum(best_f))
        best_score = compute_score(best_f)
        
    remaining_time = TIME_LIMIT - (time.time() - start_time)
    
    # Split remaining time between spectral flattening and peak shaving
    # Alternate between them to escape local minima
    
    current_f = best_f.copy()
    current_score = best_score
    
    # Iterative refinement loop
    # Use a mix of spectral flattening and peak shaving
    step_time = 5.0 # seconds per step
    
    while time.time() - start_time < TIME_LIMIT - 5:
        # Try spectral flattening
        t_start = time.time()
        f_new, score_new = spectral_flattening(
            current_f, t_start, t_start + step_time
        )
        
        if score_new < current_score:
            current_f = f_new
            current_score = score_new
            if current_score < best_score:
                best_score = current_score
                best_f = current_f.copy()
        else:
            # If spectral flattening didn't help, try peak shaving
            t_start = time.time()
            f_new, score_new = peak_shaving(
                current_f, t_start, t_start + step_time
            )
            if score_new < current_score:
                current_f = f_new
                current_score = score_new
                if current_score < best_score:
                    best_score = current_score
                    best_f = current_f.copy()
            else:
                # No improvement, break to avoid wasting time
                break
                
    # Final high-precision polish
    if time.time() - start_time < TIME_LIMIT - 2:
        t_start = time.time()
        f_polished, score_polished = peak_shaving(
            best_f, t_start, TIME_LIMIT - 2, threshold_frac=0.99
        )
        if score_polished < best_score:
            best_score = score_polished
            best_f = f_polished
    
    best_f = np.maximum(best_f, 0.0)
    return [float(x) for x in best_f]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
