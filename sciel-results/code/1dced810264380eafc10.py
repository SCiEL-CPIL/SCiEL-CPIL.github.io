"""
First autocorrelation inequality: minimize normalized max of autoconvolution
for a nonnegative step function with 600 equal intervals on [-1/4, 1/4].

Strategy:
1. Initialize with smooth Gaussian bump parametric family (fast, no slow gradients).
2. Use spectral subgradient descent: work in frequency domain where autoconvolution
   peak relates to power spectrum, apply subgradient updates on time-domain signal.
3. Apply local refinement using sensitivity-based peak reduction with FFT.
4. Final polish pass with adaptive step sizes.

Key improvements:
- Eliminated slow numerical gradient BFGS (was bottleneck)
- Spectral subgradient descent for global-ish optimization
- FFT-based autoconvolution for O(N log N)
- Fast parametric initialization without expensive optimization
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

def spectral_subgradient_descent(f_init, start_time, max_time, lr_init=0.05):
    """
    Spectral subgradient descent to minimize the peak of the autoconvolution.
    Uses the fact that for normalized f (sum=1), the autoconvolution peak is
    related to the L1 norm of the power spectrum |F|^2.
    
    Subgradient of max(conv) w.r.t. f is conv shifted to align with f.
    We use FFT to compute this efficiently.
    """
    best_f = f_init.copy()
    best_val = compute_score(f_init)
    current_f = f_init.copy()
    lr = lr_init
    M = 2048  # FFT size for linear convolution (zero padding, N=600 -> 2N-1=1199 < 2048)
    
    while time.time() - start_time < max_time:
        s = np.sum(current_f)
        if s < 1e-12:
            break
        
        # Normalize: score is scale-invariant (max(k*f * k*f) / (k*s)^2 = same)
        # Fix s = N to keep numbers stable
        current_f = current_f * (N / s)
        
        # Compute autoconvolution via FFT
        F = np.fft.rfft(current_f, n=M)
        conv = np.fft.irfft(F * F, n=M)[:2*N-1]
        max_c = np.max(conv)
        current_val = max_c / (N * N)
        
        if current_val < best_val:
            best_val = current_val
            best_f = current_f.copy()
        
        # Subgradient of max(conv) w.r.t. f:
        # conv[m] = sum_j f[j] * f[m-j]
        # d(conv[m])/df[j] = f[m-j] for the max lag m*
        # So subgradient g[j] = f[m*-j] for j in range, where m* is argmax
        # More precisely: g = ifft(2*F*phase_correction) but simpler:
        # g[j] = sum over peak lags m of f[m-j] (clamped to valid range)
        
        # Find peak lags
        threshold = max_c * 0.95
        peak_indices = np.where(conv >= threshold)[0]
        
        # Compute subgradient: for each peak lag m, contribute f[m-j] to g[j]
        g = np.zeros(N)
        for m in peak_indices:
            j_min = max(0, m - (N - 1))
            j_max = min(N - 1, m)
            if j_min <= j_max:
                # f[m-j] for j in [j_min, j_max]
                # m-j ranges from m-j_max to m-j_min
                idx = m - np.arange(j_min, j_max + 1)
                g[j_min:j_max+1] += current_f[idx]
        
        # Normalize subgradient
        g_norm = np.linalg.norm(g)
        if g_norm < 1e-15:
            break
        g = g / g_norm
        
        # Update: f_new = f - lr * g, then project to nonnegative
        f_new = current_f - lr * g
        f_new = np.maximum(f_new, 0.0)
        
        # Re-normalize
        s_new = np.sum(f_new)
        if s_new < 1e-12:
            break
        f_new = f_new * (N / s_new)
        
        # Check improvement
        F_new = np.fft.rfft(f_new, n=M)
        conv_new = np.fft.irfft(F_new * F_new, n=M)[:2*N-1]
        val_new = np.max(conv_new) / (N * N)
        
        if val_new < current_val:
            current_f = f_new
            lr = min(lr * 1.03, 0.3)
        else:
            lr *= 0.9
            if lr < 1e-10:
                break
    
    return best_f, best_val

def local_refinement(h, start_time, max_time, alpha_init=0.1, threshold_frac=0.9):
    """Local refinement using sensitivity-based peak reduction with FFT."""
    best_h = h.copy()
    best_val = compute_score(h)
    current_h = h.copy()
    alpha = alpha_init
    M = 2048
    
    while time.time() - start_time < max_time:
        s = np.sum(current_h)
        if s < 1e-12:
            break
        current_h = current_h * (N / s)
        
        F = np.fft.rfft(current_h, n=M)
        conv = np.fft.irfft(F * F, n=M)[:2*N-1]
        max_c = np.max(conv)
        current_val = max_c / (N * N)
        
        if current_val < best_val:
            best_val = current_val
            best_h = current_h.copy()
        
        # Find lags where conv is close to max
        threshold = max_c * threshold_frac
        peak_indices = np.where(conv >= threshold)[0]
        
        # Compute sensitivity
        sensitivity = np.zeros(N)
        for m in peak_indices:
            j_min = max(0, m - (N - 1))
            j_max = min(N - 1, m)
            if j_min <= j_max:
                idx = m - np.arange(j_min, j_max + 1)
                sensitivity[j_min:j_max+1] += current_h[idx]
        
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
        
        F_new = np.fft.rfft(h_new, n=M)
        conv_new = np.fft.irfft(F_new * F_new, n=M)[:2*N-1]
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
    M = 2048
    
    while time.time() - start_time < max_time:
        s = np.sum(current_h)
        if s < 1e-12:
            break
        current_h = current_h * (N / s)
        
        F = np.fft.rfft(current_h, n=M)
        conv = np.fft.irfft(F * F, n=M)[:2*N-1]
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
                idx = m - np.arange(j_min, j_max + 1)
                sensitivity[j_min:j_max+1] += current_h[idx]
        
        sens_mean = np.mean(sensitivity)
        sens_std = np.std(sensitivity) + 1e-12
        sens_norm = (sensitivity - sens_mean) / sens_std
        
        h_new = current_h * (1.0 - alpha * sens_norm)
        h_new = np.maximum(h_new, 0.0)
        
        s_new = np.sum(h_new)
        if s_new < 1e-12:
            break
        h_new = h_new * (N / s_new)
        
        F_new = np.fft.rfft(h_new, n=M)
        conv_new = np.fft.irfft(F_new * F_new, n=M)[:2*N-1]
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
    
    # Generate diverse initial candidates from Gaussian parametric family (fast, no optimization)
    x_grid = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25 / N
    
    candidate_list = []
    
    # Constant
    candidate_list.append(np.ones(N))
    
    # Gaussian bumps with various configurations
    for n_c in [2, 3, 4, 5, 6, 8]:
        if time.time() - start_time > TIME_LIMIT - 60:
            break
        
        rng = np.random.RandomState(42 + n_c)
        
        # Multiple structured initializations per n_c
        for restart in range(8):
            if restart == 0:
                centers = np.linspace(-0.2, 0.2, n_c)
                widths = np.full(n_c, 0.15)
                heights = np.full(n_c, 1.0)
            elif restart == 1:
                centers = rng.uniform(-0.22, 0.22, size=n_c)
                widths = rng.uniform(0.05, 0.25, size=n_c)
                heights = rng.uniform(0.5, 2.0, size=n_c)
            elif restart == 2:
                centers = np.zeros(n_c) + rng.uniform(-0.08, 0.08, size=n_c)
                widths = rng.uniform(0.04, 0.15, size=n_c)
                heights = rng.uniform(0.8, 3.0, size=n_c)
            elif restart == 3:
                # Alternating wide/narrow
                centers = np.linspace(-0.18, 0.18, n_c)
                widths = np.where(np.arange(n_c) % 2 == 0, 0.18, 0.06)
                heights = np.where(np.arange(n_c) % 2 == 0, 1.0, 2.0)
            elif restart == 4:
                # Two clusters
                half = n_c // 2
                centers = np.concatenate([
                    rng.uniform(-0.22, -0.1, size=half),
                    rng.uniform(0.1, 0.22, size=n_c - half)
                ])
                widths = rng.uniform(0.04, 0.12, size=n_c)
                heights = rng.uniform(1.0, 2.5, size=n_c)
            elif restart == 5:
                # Very spread out
                centers = np.linspace(-0.24, 0.24, n_c)
                widths = np.full(n_c, 0.3)
                heights = np.full(n_c, 0.8)
            elif restart == 6:
                # Center-heavy
                centers = rng.normal(0, 0.05, size=n_c)
                widths = rng.uniform(0.03, 0.1, size=n_c)
                heights = rng.uniform(1.5, 4.0, size=n_c)
            else:
                centers = rng.uniform(-0.23, 0.23, size=n_c)
                widths = rng.uniform(0.02, 0.3, size=n_c)
                heights = rng.uniform(0.3, 3.5, size=n_c)
            
            f = gaussian_basis(x_grid, centers, widths, heights)
            f = np.maximum(f, 0.0)
            if np.sum(f) > 1e-10:
                candidate_list.append(f)
    
    # Also try some simple structured patterns
    # Cosine taper
    t = np.linspace(-1, 1, N)
    candidate_list.append(np.maximum(1.0 + 0.5 * np.cos(np.pi * t), 0.0))
    # Sine bump
    candidate_list.append(np.maximum(1.0 + 0.8 * np.cos(2 * np.pi * t), 0.0))
    # Two-lobe
    candidate_list.append(np.maximum(1.0 + 1.5 * np.cos(4 * np.pi * t), 0.0))
    
    # Pick the best few candidates and run spectral subgradient descent on each
    # First, score all candidates
    scored = []
    for f in candidate_list:
        sc = compute_score(f)
        scored.append((sc, f))
    
    scored.sort(key=lambda x: x[0])
    
    # Take top K candidates for spectral optimization
    K = min(10, len(scored))
    top_candidates = [f for _, f in scored[:K]]
    
    # Update best
    if top_candidates:
        best_f = top_candidates[0].copy()
        best_score = scored[0][0]
    
    # Run spectral subgradient descent on top candidates
    for i, f_init in enumerate(top_candidates):
        if time.time() - start_time > TIME_LIMIT - 30:
            break
        
        # Allocate more time to better initial candidates
        time_budget = min(TIME_LIMIT - 15 - time.time() - start_time, 15.0)
        if time_budget < 2.0:
            break
        
        try:
            refined_f, refined_score = spectral_subgradient_descent(
                f_init, start_time, start_time + time_budget
            )
            if refined_score < best_score:
                best_f = refined_f
                best_score = refined_score
        except Exception:
            pass
    
    # Local refinement: sensitivity-based peak reduction
    if time.time() - start_time < TIME_LIMIT - 20:
        try:
            refined_f, refined_score = local_refinement(
                best_f, start_time, TIME_LIMIT - 8
            )
            if refined_score < best_score:
                best_f = refined_f
                best_score = refined_score
        except Exception:
            pass
    
    # Final polish pass
    if time.time() - start_time < TIME_LIMIT - 5:
        try:
            polished_f, polished_score = polish_pass(
                best_f, start_time, TIME_LIMIT - 2
            )
            if polished_score < best_score:
                best_f = polished_f
                best_score = polished_score
        except Exception:
            pass
    
    # Ensure nonnegative
    best_f = np.maximum(best_f, 0.0)
    return [float(x) for x in best_f]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
