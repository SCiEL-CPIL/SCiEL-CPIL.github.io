"""
First autocorrelation inequality: minimize normalized max of autoconvolution
for a nonnegative step function with 600 equal intervals on [-1/4, 1/4].

Strategy:
1. Cosine-bump parametric initialization via Nelder-Mead (gives good starting shape).
2. Spectral (frequency-domain) optimization with log-sum-exp smooth surrogate:
   - Work in the FFT domain; peak of autoconvolution = max of inverse FFT of |F|^2.
   - Gradient of smooth_max w.r.t. f is 2 * corr(weights, f), computed via FFT.
   - Use projected gradient descent / spectral updates to reduce the peak.
3. Time-domain coordinate refinement for fine-tuning.
4. Final polish pass.
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
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    n_fft = 1
    while n_fft < 2 * N - 1:
        n_fft *= 2
    F = np.fft.rfft(f, n=n_fft)
    conv = np.fft.irfft(F * F, n=n_fft)[:2*N-1]
    return np.max(conv) / (s * s)

def fft_autoconv(f, n_fft=None):
    """Compute autoconvolution using FFT."""
    if n_fft is None:
        n_fft = 1
        while n_fft < 2 * N - 1:
            n_fft *= 2
    F = np.fft.rfft(f, n=n_fft)
    conv = np.fft.irfft(F * F, n=n_fft)[:2*N-1]
    return conv

def smooth_max(x, beta):
    """Smooth approximation to max using log-sum-exp."""
    x = np.asarray(x, dtype=np.float64)
    x_max = np.max(x)
    if x_max < 1e-15:
        return 0.0
    shifted = x - x_max
    log_sum = np.log(np.sum(np.exp(beta * shifted))) / beta
    return x_max + log_sum

def smooth_max_weights(x, beta):
    """Return (smooth_max_value, softmax_weights) for gradient computation."""
    x = np.asarray(x, dtype=np.float64)
    x_max = np.max(x)
    if x_max < 1e-15:
        return 0.0, np.zeros_like(x)
    shifted = x - x_max
    exp_vals = np.exp(beta * shifted)
    log_sum = np.log(np.sum(exp_vals)) / beta
    sm_val = x_max + log_sum
    weights = exp_vals / np.sum(exp_vals)
    return sm_val, weights

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
    conv = fft_autoconv(f)
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    return smooth_max(conv, beta) / (s * s)

def gradient_of_smooth_max_conv(f, beta):
    """
    Compute gradient of smooth_max(autoconv(f), beta) / (sum(f))^2 w.r.t. f.
    
    autoconv(f)[k] = sum_i f[i]*f[k-i]
    d/d f[j] of autoconv(f)[k] = 2 * f[k-j] (when valid)
    
    Gradient of smooth_max: sum_k w_k * 2 * f[k-j] = 2 * corr(w, f)[j]
    where corr(w, f)[j] = sum_k w[k] * f[k-j]
    
    Gradient of L/S^2 = (grad_L * S^2 - L * 2 * S) / S^4 = grad_L/S^2 - 2*L/S^3
    """
    n_fft = 1
    while n_fft < 2 * N - 1:
        n_fft *= 2
    
    F = np.fft.rfft(f, n=n_fft)
    conv_full = np.fft.irfft(F * F, n=n_fft)
    conv = conv_full[:2*N-1]
    
    sm_val, weights = smooth_max_weights(conv, beta)
    
    # Gradient of smooth_max w.r.t. f[j] = 2 * sum_k weights[k] * f[k-j]
    # This is 2 * cross-correlation of weights (padded) and f
    # corr[j] = sum_k w[k] * f[k-j]
    # = sum_k w[k] * f[k-j]  (for k-j in [0, N-1])
    
    # Compute via FFT: correlation of w and f
    # corr = ifft(fft(w) * conj(fft(f)))  -- but need to be careful with padding
    # Actually: sum_k w[k] * f[k-j] for k in [0, 2N-2], j in [0, N-1]
    # Let's pad f to length 2N-1 (zeros for the rest)
    f_padded = np.zeros(2*N-1)
    f_padded[:N] = f
    # w is already length 2N-1
    
    n_corr_fft = 1
    while n_corr_fft < 2*(2*N-1):
        n_corr_fft *= 2
    
    W = np.fft.rfft(weights, n=n_corr_fft)
    Fp = np.fft.rfft(f_padded, n=n_corr_fft)
    
    # correlation[j] = sum_k w[k] * f_padded[k-j]
    # = sum_k w[k] * f_padded[k-j]
    # In frequency: corr = ifft(conj(W) * Fp)  -- standard correlation
    # Actually: (w * f)(n) = sum_k w(k) f(n-k)
    # We want sum_k w(k) f(k-j) = (f * w_reversed)(j) ... let me think again.
    # sum_k w[k] f[k-j]: let m = k-j, then k = m+j, sum_m w[m+j] f[m]
    # = sum_m w_shifted[m] f[m] where w_shifted[m] = w[m+j]
    # This is the cross-correlation of w and f at lag j.
    # corr_w_f[j] = sum_m w[m+j] f[m] = (f ⊛ w)(j) where ⊛ is cross-correlation
    # In FFT: ifft(conj(FFT(w)) * FFT(f)) gives (f ⊛ w)
    
    corr = np.fft.irfft(np.conj(W) * Fp, n=n_corr_fft)[:N]
    
    # Gradient of L = smooth_max = 2 * corr
    grad_L = 2.0 * corr
    
    s = np.sum(f)
    # Gradient of L/S^2 = grad_L/S^2 - 2*L/S^3 * 1
    grad = grad_L / (s * s) - 2.0 * sm_val / (s * s * s) * np.ones(N)
    
    return grad, sm_val, s

def spectral_optimization(f, start_time, max_time, beta_init=20.0, lr_init=0.02):
    """
    Optimize f in the spectral domain to minimize the normalized peak autoconvolution.
    Uses smooth log-sum-exp surrogate with gradient descent.
    """
    best_f = f.copy()
    best_val = compute_score(f)
    current_f = f.copy()
    beta = beta_init
    lr = lr_init
    no_improve = 0
    max_no_improve = 200
    
    n_fft = 1
    while n_fft < 2 * N - 1:
        n_fft *= 2
    
    while time.time() - start_time < max_time and no_improve < max_no_improve:
        s = np.sum(current_f)
        if s < 1e-12:
            break
        current_f = current_f * (N / s)
        
        conv = fft_autoconv(current_f, n_fft)
        max_c = np.max(conv)
        current_val = max_c / (N * N)
        
        if current_val < best_val:
            best_val = current_val
            best_f = current_f.copy()
            no_improve = 0
        else:
            no_improve += 1
        
        # Decrease beta as we converge (sharper max approximation)
        beta = max(5.0, beta_init * np.exp(-0.01 * no_improve))
        
        # Compute gradient
        grad, sm_val, s_val = gradient_of_smooth_max_conv(current_f, beta)
        
        # Gradient descent step
        f_new = current_f - lr * grad
        f_new = np.maximum(f_new, 0.0)
        
        # Renormalize
        s_new = np.sum(f_new)
        if s_new < 1e-12:
            break
        f_new = f_new * (N / s_new)
        
        # Evaluate
        conv_new = fft_autoconv(f_new, n_fft)
        val_new = np.max(conv_new) / (N * N)
        
        if val_new < current_val:
            current_f = f_new
            lr = min(lr * 1.05, 0.3)
            no_improve = 0
        else:
            # Line search: try smaller step
            improved = False
            for factor in [0.5, 0.25, 0.1, 0.05]:
                f_try = current_f - lr * factor * grad
                f_try = np.maximum(f_try, 0.0)
                s_try = np.sum(f_try)
                if s_try < 1e-12:
                    continue
                f_try = f_try * (N / s_try)
                conv_try = fft_autoconv(f_try, n_fft)
                val_try = np.max(conv_try) / (N * N)
                if val_try < current_val:
                    current_f = f_try
                    lr = min(lr * 1.02, 0.3)
                    improved = True
                    break
            if not improved:
                lr *= 0.8
                if lr < 1e-10:
                    break
    
    return best_f, best_val

def coordinate_descent(f, start_time, max_time, n_passes=5):
    """
    Coordinate descent: for each bin j, find optimal value given all others.
    The peak autoconvolution as a function of f[j] is:
    C_peak = max_k [sum_{i≠j} f[i]f[k-i] + f[j]*(f[k-j] + f[j-k] if j≠k-j else 2*f[j])]
    
    For a fixed peak index k*, C_k* as a function of f[j]:
    - If k* is even and j = k*/2: C_k* = A + 2*f[j]*f[j] + B*f[j]  (quadratic in f[j])
      where A = sum_{i≠j} f[i]f[k*-i], B = 2*f[j]  -- wait, this isn't right.
    
    Let me redo: C_k = sum_i f[i] * f[k-i]
    Terms involving f[j]:
    - i=j: f[j] * f[k-j]
    - i=k-j: f[k-j] * f[j]  (same as above if j=k-j, i.e., k=2j)
    
    So C_k = (terms not involving f[j]) + f[j]*f[k-j] + (if k-j ≠ j) f[k-j]*f[j]
    = A_k + f[j] * (f[k-j] + f[j-k])  where f[m]=0 if m out of [0,N-1]
    
    If k-j = j (i.e., k=2j): C_k = A_k + f[j]^2  (the two terms coincide)
    Actually: when k=2j, i=j and i=k-j=j are the same index, so only one term f[j]*f[j]=f[j]^2.
    When k≠2j: C_k = A_k + 2*f[j]*f[k-j]
    
    So for a given peak index k*, the constraint C_k* ≤ T gives:
    - If k*=2j: f[j]^2 ≤ T - A_k*  →  f[j] ≤ sqrt(max(0, T-A_k*))
    - If k*≠2j: 2*f[j]*f[k*-j] ≤ T - A_k*  →  f[j] ≤ (T-A_k*)/(2*f[k*-j]) if f[k*-j]>0
    
    We can binary search on T (the peak value) and for each T, set each f[j] to its max
    allowed value, then check if the resulting peak is ≤ T.
    """
    best_f = f.copy()
    best_val = compute_score(f)
    current_f = f.copy()
    
    n_fft = 1
    while n_fft < 2 * N - 1:
        n_fft *= 2
    
    for p in range(n_passes):
        if time.time() - start_time >= max_time:
            break
        
        conv = fft_autoconv(current_f, n_fft)
        current_val = np.max(conv) / (N * N)
        if current_val < best_val:
            best_val = current_val
            best_f = current_f.copy()
        
        # Find the dominant peak
        peak_idx = np.argmax(conv)
        peak_val = conv[peak_idx]
        
        # Binary search for minimum peak value achievable
        lo = 0.0
        hi = peak_val * 1.5
        for _ in range(30):
            mid = (lo + hi) / 2.0
            # For peak index peak_idx, compute max allowed f[j] for each j
            # C_peak_idx = A + f[j]*f[peak_idx-j] + (if peak_idx≠2j) f[peak_idx-j]*f[j]
            
            # Compute A_j = C_peak_idx - terms involving f[j]
            # C_peak_idx = sum_i f[i] f[peak_idx - i]
            # Terms with f[j]: i=j → f[j]*f[peak_idx-j], i=peak_idx-j → f[peak_idx-j]*f[j]
            
            f_kj = np.zeros(N)
            for j in range(N):
                kj = peak_idx - j
                if 0 <= kj < N:
                    f_kj[j] = current_f[kj]
            
            if peak_idx % 2 == 0:
                j_mid = peak_idx // 2
                if j_mid < N:
                    # C = A + f[j_mid]^2 where A = C - f[j_mid]^2
                    A = conv[peak_idx] - current_f[j_mid] ** 2
                    # For j = j_mid: f[j]^2 ≤ T - A
                    # For j ≠ j_mid: 2*f[j]*f[peak_idx-j] ≤ T - A_j
                    # A_j = C - 2*f[j]*f[peak_idx-j] = C - 2*f[j]*f_kj[j]
                    
                    # This is complex; let's do a simpler approach:
                    # Just reduce f at the peak bin and nearby
                    pass
            
            # Simpler approach: iteratively reduce bins that contribute most to the peak
            # Sensitivity: how much does reducing f[j] by delta reduce C_peak?
            # dC_peak/d f[j] = f[peak_idx - j] + (if peak_idx-j ≠ j) f[j - peak_idx]
            # = f[peak_idx - j] + f[(j - peak_idx) mod ... ] -- careful with bounds
            
            # Actually from earlier: dC_k/d f[j] = 2*f[k-j] (when k-j in [0,N-1])
            # except when k=2j: dC_k/d f[j] = 2*f[j]
            
            # So sensitivity[j] = 2 * f[peak_idx - j] if 0 ≤ peak_idx - j < N, else 0
            # (with the factor being 2*f[j] when peak_idx=2j, which is the same formula)
            
            sens = np.zeros(N)
            for j in range(N):
                kj = peak_idx - j
                if 0 <= kj < N:
                    sens[j] = 2.0 * current_f[kj]
            
            # Reduce f[j] proportionally to sensitivity
            sens_max = np.max(sens)
            if sens_max < 1e-15:
                break
            
            # Binary search step size
            for _ in range(20):
                step = (lo + hi) / 2.0
                f_new = current_f - step * sens
                f_new = np.maximum(f_new, 0.0)
                s_new = np.sum(f_new)
                if s_new < 1e-12:
                    hi = step
                    continue
                f_new = f_new * (N / s_new)
                conv_new = fft_autoconv(f_new, n_fft)
                if np.max(conv_new) / (N * N) < current_val * 0.99:
                    lo = step
                    current_f = f_new
                    current_val = np.max(conv_new) / (N * N)
                else:
                    hi = step
            
            if time.time() - start_time >= max_time:
                break
    
    return best_f, best_val

def peak_shaving(f, start_time, max_time, beta=15.0):
    """
    Iterative peak shaving: identify bins contributing to peaks, reduce them,
    renormalize, repeat. Uses smooth weighting for which bins to reduce.
    """
    best_f = f.copy()
    best_val = compute_score(f)
    current_f = f.copy()
    alpha = 0.05
    no_improve = 0
    max_no_improve = 150
    
    n_fft = 1
    while n_fft < 2 * N - 1:
        n_fft *= 2
    
    while time.time() - start_time < max_time and no_improve < max_no_improve:
        s = np.sum(current_f)
        if s < 1e-12:
            break
        current_f = current_f * (N / s)
        
        conv = fft_autoconv(current_f, n_fft)
        max_c = np.max(conv)
        current_val = max_c / (N * N)
        
        if current_val < best_val:
            best_val = current_val
            best_f = current_f.copy()
            no_improve = 0
        else:
            no_improve += 1
        
        # Adaptive beta: sharpen the peak weighting as we converge
        beta_cur = max(5.0, beta * np.exp(-0.02 * no_improve))
        
        # Get softmax weights over the convolution
        sm_val, weights = smooth_max_weights(conv, beta_cur)
        
        # Sensitivity: dC_k/d f[j] = 2*f[k-j]
        # Gradient of weighted peak: sum_k w[k] * 2*f[k-j]
        # = 2 * sum_k w[k] * f[k-j] = 2 * corr(w, f)[j]
        
        f_padded = np.zeros(2*N-1)
        f_padded[:N] = current_f
        
        n_corr_fft = 1
        while n_corr_fft < 2*(2*N-1):
            n_corr_fft *= 2
        
        W = np.fft.rfft(weights, n=n_corr_fft)
        Fp = np.fft.rfft(f_padded, n=n_corr_fft)
        corr = np.fft.irfft(np.conj(W) * Fp, n=n_corr_fft)[:N]
        
        sens = 2.0 * corr
        sens_max = np.max(sens)
        if sens_max < 1e-15:
            break
        
        sens_norm = sens / (sens_max + 1e-15)
        
        # Only modify bins with significant contribution
        mask = sens_norm > 0.05
        if not np.any(mask):
            break
        
        f_new = current_f.copy()
        f_new[mask] = current_f[mask] * (1.0 - alpha * sens_norm[mask])
        f_new = np.maximum(f_new, 0.0)
        
        s_new = np.sum(f_new)
        if s_new < 1e-12:
            break
        f_new = f_new * (N / s_new)
        
        conv_new = fft_autoconv(f_new, n_fft)
        val_new = np.max(conv_new) / (N * N)
        
        if val_new < current_val:
            current_f = f_new
            alpha = min(alpha * 1.03, 0.3)
            no_improve = 0
        else:
            # Line search
            improved = False
            for factor in [0.5, 0.25, 0.1]:
                f_try = current_f.copy()
                f_try[mask] = current_f[mask] * (1.0 - alpha * factor * sens_norm[mask])
                f_try = np.maximum(f_try, 0.0)
                s_try = np.sum(f_try)
                if s_try < 1e-12:
                    continue
                f_try = f_try * (N / s_try)
                conv_try = fft_autoconv(f_try, n_fft)
                val_try = np.max(conv_try) / (N * N)
                if val_try < current_val:
                    current_f = f_try
                    alpha = min(alpha * 1.02, 0.3)
                    improved = True
                    break
            if not improved:
                alpha *= 0.85
                if alpha < 1e-10:
                    break
    
    return best_f, best_val

def discrete_annealing(h, start_time, max_time):
    """Block-based annealing for fine-tuning."""
    best_h = h.copy()
    best_val = compute_score(h)
    current_h = h.copy()
    current_val = best_val
    
    s = np.sum(current_h)
    if s > 1e-12:
        current_h = current_h * (N / s)
    
    T = 0.3
    cooling = 0.997
    rng = np.random.RandomState(42)
    block_sizes = [1, 2, 3, 5]
    n_fft = 1
    while n_fft < 2 * N - 1:
        n_fft *= 2
    
    while time.time() - start_time < max_time and T > 0.001:
        bs = block_sizes[rng.randint(len(block_sizes))]
        j = rng.randint(N - bs + 1)
        
        old_block = current_h[j:j+bs].copy()
        scale = 0.15 * np.maximum(np.mean(old_block), 0.1)
        perturbation = rng.normal(0, scale, size=bs)
        new_block = np.maximum(0.0, old_block + perturbation)
        
        if np.allclose(new_block, old_block):
            continue
        
        current_h[j:j+bs] = new_block
        s_new = np.sum(current_h)
        if s_new < 1e-12:
            current_h[j:j+bs] = old_block
            continue
        current_h = current_h * (N / s_new)
        
        conv_new = fft_autoconv(current_h, n_fft)
        val_new = np.max(conv_new) / (N * N)
        
        accept = val_new < current_val
        if not accept and T > 0.01:
            accept = rng.random() < np.exp(-(val_new - current_val) / T)
        
        if accept:
            current_val = val_new
            if val_new < best_val:
                best_val = val_new
                best_h = current_h.copy()
        else:
            current_h[j:j+bs] = old_block
            s_rej = np.sum(current_h)
            if s_rej > 1e-12:
                current_h = current_h * (N / s_rej)
        
        T *= cooling
    
    return best_h, best_val

def construct():
    start_time = time.time()
    
    best_f = np.ones(N)
    best_score = compute_score(best_f)
    
    # Phase 1: Parametric initialization with cosine bumps
    for n_c in [3, 4, 5, 6]:
        if time.time() - start_time > TIME_LIMIT - 40:
            break
        
        rng = np.random.RandomState(42 + n_c)
        best_params = None
        best_obj = np.inf
        
        betas = [5.0, 10.0, 20.0]
        
        for beta in betas:
            if time.time() - start_time > TIME_LIMIT - 30:
                break
            
            for restart in range(5):
                if time.time() - start_time > TIME_LIMIT - 25:
                    break
                
                if restart == 0:
                    centers = np.linspace(-0.15, 0.15, n_c)
                    widths = np.full(n_c, 0.15)
                    heights = np.full(n_c, 1.0)
                elif restart == 1:
                    centers = rng.uniform(-0.2, 0.2, size=n_c)
                    widths = rng.uniform(0.05, 0.25, size=n_c)
                    heights = rng.uniform(0.5, 2.0, size=n_c)
                else:
                    centers = rng.uniform(-0.2, 0.2, size=n_c)
                    widths = rng.uniform(0.03, 0.3, size=n_c)
                    heights = rng.uniform(0.3, 2.5, size=n_c)
                
                x0 = np.concatenate([centers, widths, heights])
                
                try:
                    result = minimize(
                        objective, x0, args=(n_c, beta),
                        method='Nelder-Mead',
                        options={'maxiter': 1500, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True}
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
    
    # Phase 2: Spectral optimization (primary refinement)
    remaining = TIME_LIMIT - (time.time() - start_time)
    if remaining > 20:
        spectral_time = min(remaining * 0.4, 45)
        spec_f, spec_score = spectral_optimization(
            best_f, time.time(), spectral_time
        )
        if spec_score < best_score:
            best_f = spec_f
            best_score = spec_score
    
    # Phase 3: Peak shaving (smooth sensitivity-based)
    remaining = TIME_LIMIT - (time.time() - start_time)
    if remaining > 15:
        shave_time = min(remaining * 0.4, 35)
        shaved_f, shaved_score = peak_shaving(
            best_f, time.time(), shave_time
        )
        if shaved_score < best_score:
            best_f = shaved_f
            best_score = shaved_score
    
    # Phase 4: Discrete annealing for fine-tuning
    remaining = TIME_LIMIT - (time.time() - start_time)
    if remaining > 10:
        anneal_time = min(remaining * 0.4, 25)
        annealed_f, annealed_score = discrete_annealing(
            best_f, time.time(), anneal_time
        )
        if annealed_score < best_score:
            best_f = annealed_f
            best_score = annealed_score
    
    # Phase 5: Final polish (another round of peak shaving with tighter params)
    remaining = TIME_LIMIT - (time.time() - start_time)
    if remaining > 5:
        polish_time = min(remaining * 0.7, 15)
        polished_f, polished_score = peak_shaving(
            best_f, time.time(), polish_time, beta=30.0
        )
        if polished_score < best_score:
            best_f = polished_f
            best_score = polished_score
    
    best_f = np.maximum(best_f, 0.0)
    return [float(x) for x in best_f]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
