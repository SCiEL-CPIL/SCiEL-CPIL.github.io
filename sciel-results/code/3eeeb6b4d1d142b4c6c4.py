"""
First autocorrelation inequality: minimize normalized max of autoconvolution
for a nonnegative step function with 600 equal intervals on [-1/4, 1/4].

Strategy:
1. Direct optimization over the 600 bin heights (no parametric family).
2. Use FFT to compute autoconvolution quickly.
3. Use a smooth max (log-sum-exp) surrogate for the peak autoconvolution.
4. Projected gradient descent with adaptive step size and normalization.
5. Coordinate descent polish on the active peak lags to fine-tune individual bins.
"""

import numpy as np
import time

N = 600
DT = 0.5 / N  # interval width
TIME_LIMIT = 140  # safety margin before 150s kill


def autoconvolve_fft(f):
    """Compute autoconvolution using FFT."""
    n = 2 * N
    F = np.fft.rfft(f, n=n)
    conv = np.fft.irfft(F * F, n=n)[:2 * N - 1]
    return conv


def compute_score(f):
    """Compute the normalized maximum of the autoconvolution."""
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return np.inf
    f = np.maximum(f, 0.0)
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    conv = autoconvolve_fft(f)
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


def objective_and_grad(f, beta):
    """Compute objective and gradient for smooth max of autoconvolution.
    
    Returns (obj, grad) where grad is the gradient of obj w.r.t. f.
    """
    s = np.sum(f)
    if s < 1e-15:
        return np.inf, np.zeros_like(f)
    
    conv = autoconvolve_fft(f)
    obj_raw = smooth_max(conv, beta)
    obj = obj_raw / (s * s)
    
    # Gradient of smooth max
    x_max = np.max(conv)
    if x_max < 1e-15:
        grad_conv = np.zeros_like(conv)
    else:
        x_scaled = conv / x_max
        shifted = x_scaled - np.max(x_scaled)
        weights = np.exp(beta * shifted)
        weights /= np.sum(weights)
        grad_conv = weights  # d(smooth_max)/d(conv) = weights
    
    # d(obj)/d(conv[m]) = grad_conv[m] / (s*s)
    # d(conv[m])/d(f[j]) = f[m-j] (with appropriate bounds)
    # So d(obj)/d(f[j]) = (1/(s*s)) * sum_m grad_conv[m] * f[m-j]
    # This is a cross-correlation of grad_conv with f
    
    # Also need to account for d(obj)/ds from the 1/(s*s) factor
    # obj = obj_raw / s^2
    # d(obj)/d(f[j]) = (1/s^2) * d(obj_raw)/d(f[j]) + obj_raw * d(1/s^2)/d(f[j])
    # d(1/s^2)/d(f[j]) = -2/s^3
    # d(obj_raw)/d(f[j]) = sum_m grad_conv[m] * f[m-j]
    
    # Compute d(obj_raw)/d(f) via cross-correlation
    # conv[m] = sum_j f[j] * f[m-j]
    # d(conv[m])/d(f[j]) = f[m-j]
    # d(obj_raw)/d(f[j]) = sum_m grad_conv[m] * f[m-j]
    
    # This is the cross-correlation of grad_conv and f
    # Let's compute it using FFT
    n = 2 * N
    G = np.fft.rfft(grad_conv, n=n)
    F = np.fft.rfft(f, n=n)
    # cross-correlation: sum_m grad_conv[m] * f[m-j]
    # This is like (grad_conv * f) but with one flipped
    # Actually, let's think: sum_m A[m] * B[m-j] = sum_k A[k+j] * B[k]
    # This is the cross-correlation of A and B at lag j
    
    # Using FFT: cross_corr[j] = sum_m A[m] * B[m-j]
    # = sum_m A[m] * B[m-j]
    # Let k = m-j, then m = k+j
    # = sum_k A[k+j] * B[k]
    
    # This is the standard cross-correlation.
    # In FFT: IFFT(conj(A) * B) gives sum_k A[k] * B[k+j] (correlation)
    # We want sum_k A[k+j] * B[k] = sum_k B[k] * A[k+j]
    # This is IFFT(A * conj(B)) ... let's verify
    
    # Actually, let's just do it directly for clarity, but N=600 so 1200 lags * 600 bins = 720k ops, fast enough
    # But let's use FFT for elegance
    # sum_m grad_conv[m] * f[m-j] for j in 0..N-1
    # This is the cross-correlation of grad_conv and f at lags 0..N-1
    
    # Let's compute full cross-correlation
    G_full = np.fft.rfft(grad_conv, n=n)
    F_full = np.fft.rfft(f, n=n)
    # cross_corr = IFFT(conj(G_full) * F_full)
    cross_corr = np.fft.irfft(np.conj(G_full) * F_full, n=n)
    # cross_corr[j] = sum_m grad_conv[m] * f[m-j] for appropriate j
    # We need indices 0 to N-1
    d_obj_raw_df = cross_corr[:N]
    
    # Now combine
    grad = d_obj_raw_df / (s * s) - 2 * obj_raw * d_obj_raw_df / (s * s * s)
    # Wait, that's not right. Let me redo.
    # obj = obj_raw * s^(-2)
    # d(obj)/d(f[j]) = d(obj_raw)/d(f[j]) * s^(-2) + obj_raw * d(s^(-2))/d(f[j])
    # d(s^(-2))/d(f[j]) = -2 * s^(-3)
    # So grad = d_obj_raw_df / s^2 - 2 * obj_raw * (1/s^3)
    # But (1/s^3) is a scalar, and d(s)/d(f[j]) = 1, so d(s^(-2))/d(f[j]) = -2/s^3
    # So grad[j] = d_obj_raw_df[j] / s^2 - 2 * obj_raw / s^3
    
    grad = d_obj_raw_df / (s * s) - 2.0 * obj_raw / (s ** 3)
    
    return obj, grad


def project_nonnegative(f):
    """Project onto nonnegative orthant."""
    return np.maximum(f, 0.0)


def normalize_to_sum(f, target_sum):
    """Normalize f to have sum = target_sum."""
    s = np.sum(f)
    if s < 1e-15:
        return np.full_like(f, target_sum / N)
    return f * (target_sum / s)


def gradient_descent(f0, start_time, max_time, beta_schedule):
    """Run projected gradient descent with adaptive step size."""
    f = f0.copy()
    best_f = f.copy()
    best_val = compute_score(f)
    
    # Normalize to sum = N for stability
    f = normalize_to_sum(f, N)
    
    step_size = 0.1
    min_step = 1e-6
    max_step = 1.0
    patience = 0
    max_patience = 100
    
    for beta in beta_schedule:
        if time.time() - start_time > max_time:
            break
        
        # Run several iterations for this beta
        iters_for_beta = 500
        for it in range(iters_for_beta):
            if time.time() - start_time > max_time:
                break
            
            obj, grad = objective_and_grad(f, beta)
            if not np.isfinite(obj):
                break
            
            # Project gradient
            f_new = f - step_size * grad
            f_new = project_nonnegative(f_new)
            f_new = normalize_to_sum(f_new, N)
            
            new_val = compute_score(f_new)
            
            if new_val < best_val:
                best_val = new_val
                best_f = f_new.copy()
            
            if new_val < obj:
                f = f_new
                patience = 0
                step_size = min(step_size * 1.05, max_step)
            else:
                patience += 1
                step_size *= 0.8
                if step_size < min_step:
                    break
                if patience > max_patience:
                    break
    
    return best_f, best_val


def coordinate_descent(f, start_time, max_time):
    """Coordinate descent polish on active peak lags."""
    f = f.copy()
    best_f = f.copy()
    best_val = compute_score(f)
    
    f = normalize_to_sum(f, N)
    
    # Precompute some things
    conv = autoconvolve_fft(f)
    max_c = np.max(conv)
    current_val = max_c / (N * N)
    
    if current_val < best_val:
        best_val = current_val
        best_f = f.copy()
    
    # Find active peak lags
    threshold_frac = 0.95
    threshold = max_c * threshold_frac
    peak_indices = np.where(conv >= threshold)[0]
    
    # For each bin, compute its contribution to peak lags
    # We'll iterate over bins and try small perturbations
    
    alpha = 0.01
    for it in range(1000):
        if time.time() - start_time > max_time:
            break
        
        # Recompute conv and peaks
        conv = autoconvolve_fft(f)
        max_c = np.max(conv)
        current_val = max_c / (N * N)
        
        if current_val < best_val:
            best_val = current_val
            best_f = f.copy()
        
        threshold = max_c * threshold_frac
        peak_indices = np.where(conv >= threshold)[0]
        
        # Compute sensitivity for each bin
        # d(conv[m])/d(f[j]) = f[m-j]
        sensitivity = np.zeros(N)
        for m in peak_indices:
            j_min = max(0, m - (N - 1))
            j_max = min(N - 1, m)
            for j in range(j_min, j_max + 1):
                sensitivity[j] += f[m - j]
        
        # Normalize sensitivity
        sens_mean = np.mean(sensitivity)
        sens_std = np.std(sensitivity) + 1e-12
        sens_norm = (sensitivity - sens_mean) / sens_std
        
        # Try reducing high-sensitivity bins and increasing low-sensitivity bins
        f_new = f * (1.0 - alpha * sens_norm)
        f_new = project_nonnegative(f_new)
        f_new = normalize_to_sum(f_new, N)
        
        new_val = compute_score(f_new)
        
        if new_val < current_val:
            f = f_new
            alpha = min(alpha * 1.02, 0.1)
        else:
            alpha *= 0.9
            if alpha < 1e-8:
                break
    
    return best_f, best_val


def construct():
    start_time = time.time()
    
    # Initial candidates
    candidates = []
    
    # 1. Constant
    candidates.append(np.ones(N))
    
    # 2. Cosine bump
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25 / N
    f_cos = np.cos(np.pi * x / 0.25) ** 2
    f_cos = np.maximum(f_cos, 0)
    candidates.append(f_cos)
    
    # 3. Gaussian
    f_gauss = np.exp(-10 * (x / 0.25) ** 2)
    candidates.append(f_gauss)
    
    # 4. Random smooth
    rng = np.random.RandomState(42)
    f_rand = rng.random(N)
    # Smooth it
    kernel = np.ones(5) / 5
    f_rand = np.convolve(f_rand, kernel, mode='same')
    candidates.append(f_rand)
    
    # 5. Two bumps
    f_two = np.zeros(N)
    centers = [-0.1, 0.1]
    for c in centers:
        mask = np.abs(x - c) <= 0.08
        f_two[mask] += np.cos(np.pi * (x[mask] - c) / 0.16) ** 2
    candidates.append(f_two)
    
    # Evaluate all candidates
    best_f = None
    best_score = np.inf
    for f in candidates:
        f = normalize_to_sum(f, N)
        score = compute_score(f)
        if score < best_score:
            best_score = score
            best_f = f.copy()
    
    # Gradient descent with annealed beta
    if time.time() - start_time < TIME_LIMIT - 20:
        beta_schedule = [5.0, 10.0, 20.0, 50.0]
        refined_f, refined_score = gradient_descent(
            best_f, start_time, TIME_LIMIT - 10, beta_schedule
        )
        if refined_score < best_score:
            best_f = refined_f
            best_score = refined_score
    
    # Coordinate descent polish
    if time.time() - start_time < TIME_LIMIT - 5:
        polished_f, polished_score = coordinate_descent(
            best_f, start_time, TIME_LIMIT - 2
        )
        if polished_score < best_score:
            best_f = polished_f
            best_score = polished_score
    
    # Final normalization and cleanup
    best_f = np.maximum(best_f, 0.0)
    best_f = normalize_to_sum(best_f, N)
    
    return [float(x) for x in best_f]


if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
