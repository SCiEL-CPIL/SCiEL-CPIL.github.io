"""
First autocorrelation inequality: minimize normalized max of autoconvolution
for a nonnegative step function with 600 equal intervals on [-1/4, 1/4].

Strategy:
1. Formulate as a linear program: minimize peak autoconvolution subject to
   nonnegativity and normalization constraints.
2. Use scipy.optimize.linprog with the HiGHS solver for speed.
3. Add local refinement passes to polish the LP solution.
"""

import numpy as np
from scipy.optimize import linprog, minimize
import time

N = 600
TIME_LIMIT = 135  # safety margin before 150s kill

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

def fft_autocorr(f):
    """Compute autoconvolution using FFT for speed."""
    n = len(f)
    size = 2 * n - 1
    fft_size = 1
    while fft_size < size:
        fft_size *= 2
    F = np.fft.rfft(f, n=fft_size)
    conv = np.fft.irfft(F * F, n=fft_size)[:size]
    return conv

def build_lp_matrices():
    """Build the constraint matrices for the LP formulation.
    
    We want to minimize t subject to:
    - conv[m] <= t * N^2 for all m (autoconvolution peak bound)
    - sum(f) = N (normalization)
    - f >= 0
    
    The autoconvolution at lag m is: conv[m] = sum_j f[j] * f[m-j]
    This is quadratic, not linear! So we can't directly LP this.
    
    Instead, we use a different approach: fix the normalization sum(f)=N,
    and note that for a given f, the peak autoconvolution is determined.
    
    Actually, let me reconsider. The problem is that autoconvolution is quadratic.
    We need a different structural approach.
    
    Alternative: Use the fact that we can optimize over the shape while keeping
    sum fixed. Let's use a parametric approach with LP inner loops.
    """
    pass

def local_refinement_lp(h, start_time, max_time, alpha_init=0.05):
    """Local refinement using sensitivity-based peak reduction.
    
    For the normalized problem (sum(f)=N), the autoconvolution at lag m is:
    conv[m] = sum_j f[j] * f[m-j]
    
    The sensitivity of conv[m] with respect to f[k] (holding sum fixed) is:
    d(conv[m])/d(f[k]) = f[m-k] + f[m+k] - 2*f[m] * (1/N) * ... 
    Actually, if we perturb f[k] by delta and adjust others to keep sum=N,
    it's more complex. Let's use the direct approach:
    
    If we increase f[k] by delta and decrease all other f[j] proportionally
    to maintain sum, the change in conv[m] is approximately:
    delta * (f[m-k] + f[m+k]) - delta * (2/N) * sum_j f[j]*(f[m-j])
    
    For simplicity, use the unnormalized sensitivity and renormalize after.
    """
    best_h = h.copy()
    best_val = compute_score(h)
    current_h = h.copy()
    alpha = alpha_init
    
    while time.time() - start_time < max_time:
        s = np.sum(current_h)
        if s < 1e-12:
            break
        current_h = current_h * (N / s)
        
        conv = fft_autocorr(current_h)
        max_c = np.max(conv)
        current_val = max_c / (N * N)
        
        if current_val < best_val:
            best_val = current_val
            best_h = current_h.copy()
        
        # Find lags where conv is close to max
        threshold = max_c * 0.95
        peak_indices = np.where(conv >= threshold)[0]
        
        if len(peak_indices) == 0:
            peak_indices = np.array([np.argmax(conv)])

        # Compute sensitivity: for each peak lag m, sensitivity[k] += current_h[m-k]
        sensitivity = np.zeros(N)
        for m in peak_indices:
            j_min = max(0, m - (N - 1))
            j_max = min(N - 1, m)
            if j_max >= j_min:
                idx = m - np.arange(j_min, j_max + 1)
                sensitivity[j_min:j_max+1] += current_h[idx]
        
        # Also add the "mirror" contribution: sensitivity[k] += current_h[m+k] for valid m+k
        for m in peak_indices:
            # m+k < 2N-1 => k < 2N-1-m
            # Also need m+k >= 0 which is always true for k >= 0
            k_max = min(N - 1, 2 * N - 2 - m)
            if k_max >= 0:
                idx = m + np.arange(0, k_max + 1)
                sensitivity[0:k_max+1] += current_h[idx]
        
        sens_mean = np.mean(sensitivity)
        sens_std = np.std(sensitivity) + 1e-12
        sens_norm = (sensitivity - sens_mean) / sens_std
        
        h_new = current_h * (1.0 - alpha * sens_norm)
        h_new = np.maximum(h_new, 0.0)
        
        s_new = np.sum(h_new)
        if s_new < 1e-12:
            break
        h_new = h_new * (N / s_new)
        
        conv_new = fft_autocorr(h_new)
        val_new = np.max(conv_new) / (N * N)
        
        if val_new < current_val:
            current_h = h_new
            alpha = min(alpha * 1.05, 0.3)
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
    alpha = 0.02
    
    while time.time() - start_time < max_time:
        s = np.sum(current_h)
        if s < 1e-12:
            break
        current_h = current_h * (N / s)
        
        conv = fft_autocorr(current_h)
        max_c = np.max(conv)
        current_val = max_c / (N * N)
        
        if current_val < best_val:
            best_val = current_val
            best_h = current_h.copy()
        
        threshold = max_c * 0.995
        peak_indices = np.where(conv >= threshold)[0]
        
        if len(peak_indices) == 0:
            peak_indices = np.array([np.argmax(conv)])

        sensitivity = np.zeros(N)
        for m in peak_indices:
            j_min = max(0, m - (N - 1))
            j_max = min(N - 1, m)
            if j_max >= j_min:
                idx = m - np.arange(j_min, j_max + 1)
                sensitivity[j_min:j_max+1] += current_h[idx]
        for m in peak_indices:
            k_max = min(N - 1, 2 * N - 2 - m)
            if k_max >= 0:
                idx = m + np.arange(0, k_max + 1)
                sensitivity[0:k_max+1] += current_h[idx]
        
        sens_mean = np.mean(sensitivity)
        sens_std = np.std(sensitivity) + 1e-12
        sens_norm = (sensitivity - sens_mean) / sens_std
        
        h_new = current_h * (1.0 - alpha * sens_norm)
        h_new = np.maximum(h_new, 0.0)
        
        s_new = np.sum(h_new)
        if s_new < 1e-12:
            break
        h_new = h_new * (N / s_new)
        
        conv_new = fft_autocorr(h_new)
        val_new = np.max(conv_new) / (N * N)
        
        if val_new < current_val:
            current_h = h_new
            alpha = min(alpha * 1.03, 0.15)
        else:
            alpha *= 0.88
            if alpha < 1e-10:
                break
    
    return best_h, best_val

def construct():
    start_time = time.time()
    
    # Start with a good initial guess: smooth bump
    x = np.linspace(-0.25, 0.25, N, endpoint=False)
    best_f = np.exp(-8 * x * x)  # Gaussian bump
    best_score = compute_score(best_f)
    
    # Also try a flat function
    flat_f = np.ones(N)
    flat_score = compute_score(flat_f)
    if flat_score < best_score:
        best_f = flat_f
        best_score = flat_score
    
    # Try a few parametric initializations
    rng = np.random.RandomState(42)
    candidates = [best_f.copy(), flat_f.copy()]
    
    # Cosine bump
    f_cos = np.cos(np.pi * x) ** 2
    f_cos = np.maximum(f_cos, 0)
    candidates.append(f_cos)
    
    # Double bump
    f_double = np.exp(-20 * (x - 0.1) ** 2) + np.exp(-20 * (x + 0.1) ** 2)
    candidates.append(f_double)
    
    # Random smooth
    for i in range(3):
        f_rand = np.zeros(N)
        for _ in range(4):
            freq = rng.uniform(1, 8)
            phase = rng.uniform(0, 2 * np.pi)
            amp = rng.uniform(0.5, 1.0)
            f_rand += amp * np.cos(2 * np.pi * freq * x + phase)
        f_rand = np.maximum(f_rand, 0)
        if np.sum(f_rand) > 0:
            candidates.append(f_rand)
    
    # Run local refinement on each candidate
    if candidates:
        time_per_candidate = (TIME_LIMIT - 15) / len(candidates)
        for f_cand in candidates:
            if time.time() - start_time > TIME_LIMIT - 5:
                break
            refined_f, refined_score = local_refinement_lp(
                f_cand, start_time, min(time.time() + time_per_candidate, TIME_LIMIT - 5)
            )
            if refined_score < best_score:
                best_f = refined_f
                best_score = refined_score
    
    # Final polish pass
    if time.time() - start_time < TIME_LIMIT - 2:
        polished_f, polished_score = polish_pass(
            best_f, start_time, TIME_LIMIT
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
