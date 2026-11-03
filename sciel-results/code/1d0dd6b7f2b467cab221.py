"""
First autocorrelation inequality: minimize normalized max of autoconvolution
for a nonnegative step function with 600 equal intervals on [-1/4, 1/4].

Strategy (STRUCTURAL CHANGE):
Formulate as convex optimization in spectral domain:
- Autoconvolution peak is bounded by L1 norm of power spectrum |F|^2
- Minimize ||F||_1^2 via projected gradient descent on frequency magnitudes
- Projection enforces non-negativity of inverse DFT via soft-thresholding
- Use multiple restarts with different spectral initializations
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

def gaussian_basis(x, centers, widths, heights):
    """Generate a smooth nonnegative function from Gaussian bump components."""
    result = np.zeros_like(x)
    for c, w, h in zip(centers, widths, heights):
        result += h * np.exp(-0.5 * ((x - c) / (w / 2.0)) ** 2)
    return result

def project_nonnegative(f):
    """Project onto nonnegative cone with normalization."""
    f = np.maximum(f, 0.0)
    s = np.sum(f)
    if s < 1e-15:
        return f
    return f * (N / s)

def spectral_pgd_optimize(f_init, start_time, deadline, n_iters=500, lr_init=0.01):
    """
    Projected Gradient Descent on spectral magnitudes.
    
    Objective: minimize max of autoconvolution, approximated by 
    minimizing L1 norm of |F|^2 (power spectrum).
    
    The gradient of |||F|^2||_1 w.r.t. frequency magnitudes is computed
    analytically. Projection enforces non-negativity of time-domain signal.
    """
    current_f = project_nonnegative(f_init.copy())
    F = np.fft.fft(current_f)
    mags = np.abs(F)
    phases = np.angle(F)
    
    best_f = current_f.copy()
    best_val = compute_score(current_f)
    
    lr = lr_init
    n_half = N // 2
    
    # Precompute which bins are conjugate pairs
    # For real signal: F[N-k] = conj(F[k])
    # We optimize mags[0..n_half] and mirror
    
    for it in range(n_iters):
        if time.time() > deadline:
            break
        
        # Compute power spectrum P = |F|^2
        P = mags ** 2
        
        # Gradient of L1(P) w.r.t. mags:
        # d/dm_k sum_j P_j = 2 * m_k * (1 if k in support else 0)
        # For L1, subgradient is sign(P_k) * 2 * m_k
        # We want to minimize, so gradient direction is +2*m_k*sign(P_k)
        # Since P_k >= 0, sign is +1, so grad = 2*m_k
        
        # But we need gradient w.r.t. the actual optimization variable.
        # Let's use: minimize sum_k |F_k|^2 = sum_k mags_k^2
        # Gradient w.r.t. mags_k = 2 * mags_k
        
        # However, the true objective is max of autoconvolution, not L1 of power.
        # L1 of power is an upper bound. Let's use it as a surrogate.
        
        grad = 2.0 * mags
        
        # Update magnitudes
        mags_new = mags - lr * grad
        
        # Enforce non-negative magnitudes
        mags_new = np.maximum(mags_new, 0.0)
        
        # Reconstruct F
        F_new = mags_new * np.exp(1j * phases)
        
        # Ensure Hermitian symmetry for real signal
        for k in range(1, n_half + 1):
            F_new[N - k] = np.conj(F_new[k])
        
        # Inverse FFT
        f_new = np.real(np.fft.ifft(F_new))
        
        # Project onto nonnegative cone
        f_new = project_nonnegative(f_new)
        
        # Update mags and phases from projected signal
        F_proj = np.fft.fft(f_new)
        mags = np.abs(F_proj)
        phases = np.angle(F_proj)
        
        # Evaluate
        val = compute_score(f_new)
        if val < best_val:
            best_val = val
            best_f = f_new.copy()
        
        # Adaptive learning rate
        lr *= 0.995
    
    return best_f, best_val

def spectral_search(f_init, start_time, deadline):
    """
    Multi-restart spectral search with different initializations.
    """
    best_f = f_init.copy()
    best_val = compute_score(f_init)
    
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25 / N
    
    # Generate diverse initializations
    inits = []
    
    # 1. Gaussian bumps
    for n_c in [3, 5, 7]:
        rng = np.random.RandomState(100 + n_c)
        centers = rng.uniform(-0.2, 0.2, size=n_c)
        widths = rng.uniform(0.05, 0.2, size=n_c)
        heights = rng.uniform(0.5, 2.0, size=n_c)
        f = gaussian_basis(x, centers, widths, heights)
        f = project_nonnegative(f)
        inits.append(f)
    
    # 2. Flat top with sinusoidal modulation
    for freq in [1, 2, 3, 5]:
        f = 1.0 + 0.5 * np.sin(2 * np.pi * freq * x / 0.5)
        f = project_nonnegative(f)
        inits.append(f)
    
    # 3. Cosine window
    f = 0.5 * (1 + np.cos(2 * np.pi * x / 0.5))
    f = project_nonnegative(f)
    inits.append(f)
    
    # 4. Triangular
    f = 1.0 - np.abs(x) / 0.25
    f = project_nonnegative(f)
    inits.append(f)
    
    # 5. Original
    inits.append(project_nonnegative(f_init.copy()))
    
    # Run spectral PGD on each initialization
    for i, f_init_i in enumerate(inits):
        if time.time() > deadline - 5:
            break
        
        f_opt, val = spectral_pgd_optimize(f_init_i, start_time, deadline - 5)
        if val < best_val:
            best_val = val
            best_f = f_opt.copy()
    
    return best_f, best_val

def local_refinement(h, start_time, deadline, alpha_init=0.1):
    """Local refinement using sensitivity-based peak reduction with FFT."""
    best_h = h.copy()
    best_val = compute_score(h)
    current_h = project_nonnegative(h.copy())
    alpha = alpha_init
    
    while time.time() < deadline:
        conv = fftconvolve(current_h, current_h, mode='full')
        max_c = np.max(conv)
        current_val = max_c / (N * N)
        
        if current_val < best_val:
            best_val = current_val
            best_h = current_h.copy()
        
        # Find lags where conv is close to max
        threshold = max_c * 0.9
        peak_indices = np.where(conv >= threshold)[0]
        
        # Compute sensitivity
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
        h_new = project_nonnegative(h_new)
        
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

def construct():
    start_time = time.time()
    
    # Initial best: constant function
    best_f = np.ones(N)
    best_score = compute_score(best_f)
    
    # Spectral search with multiple initializations
    if time.time() < TIME_LIMIT - 20:
        spec_f, spec_score = spectral_search(best_f, start_time, TIME_LIMIT - 15)
        if spec_score < best_score:
            best_f = spec_f
            best_score = spec_score
    
    # Local refinement
    if time.time() < TIME_LIMIT - 5:
        refined_f, refined_score = local_refinement(best_f, start_time, TIME_LIMIT - 2)
        if refined_score < best_score:
            best_f = refined_f
            best_score = refined_score
    
    # Ensure nonnegative
    best_f = np.maximum(best_f, 0.0)
    return [float(x) for x in best_f]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
