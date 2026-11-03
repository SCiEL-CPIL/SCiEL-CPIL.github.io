"""
First autocorrelation inequality: minimize normalized max of autoconvolution
for a nonnegative step function with 600 equal intervals on [-1/4, 1/4].

Strategy: 
1. Initialize using a smooth parametric family based on cosine bumps.
2. Optimize using a sensitivity-based peak-shaving algorithm.
   - This method identifies lags where the autoconvolution is near its maximum.
   - It computes the sensitivity of each input bin to these peak lags.
   - It reduces bins that contribute most to the peaks, effectively flattening the autoconvolution.
3. Apply a final high-precision polish pass with tighter tolerances and smaller steps.

This approach is significantly faster and more effective for "flattening" objectives 
than coordinate descent, as it directly targets the maximum value.
"""

import numpy as np
import time

N = 600
TIME_LIMIT = 145.0  # Safety margin before 150s kill

def compute_score(f):
    """
    Compute the normalized maximum of the autoconvolution.
    Score = max(conv(f, f)) / (sum(f))^2
    """
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return np.inf
    f = np.maximum(f, 0.0)
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    # Use full convolution
    conv = np.convolve(f, f, mode='full')
    max_conv = np.max(conv)
    return max_conv / (s * s)

def generate_cosine_initial(n_components, rng):
    """
    Generate an initial step function using a sum of cosine bumps.
    This provides a structured, non-trivial starting point.
    """
    x = np.linspace(0, N, N, endpoint=False)
    # Normalize x to [0, 1] for parameter interpretation
    x_norm = x / N
    
    centers = rng.uniform(0.2, 0.8, size=n_components)
    widths = rng.uniform(0.1, 0.3, size=n_components)
    heights = rng.uniform(0.5, 1.5, size=n_components)
    
    f = np.zeros(N)
    for c, w, h in zip(centers, widths, heights):
        # Cosine bump: h * cos^2(pi * (x - c) / w) for |x - c| < w/2
        dist = np.abs(x_norm - c)
        mask = dist < (w / 2.0)
        if np.any(mask):
            f[mask] += h * np.cos(np.pi * (x_norm[mask] - c) / w) ** 2
            
    # Ensure non-negative and normalize to sum N
    f = np.maximum(f, 0.0)
    s = np.sum(f)
    if s > 1e-10:
        f = f * (N / s)
    else:
        f = np.ones(N)
        
    return f

def peak_shaving_step(f, alpha, peak_threshold=0.95):
    """
    Perform one step of sensitivity-based peak shaving.
    
    1. Compute autoconvolution.
    2. Identify peak lags.
    3. Compute sensitivity of peaks to each bin.
    4. Update bins by reducing those with high sensitivity.
    """
    f = np.asarray(f, dtype=np.float64)
    f = np.maximum(f, 0.0)
    
    conv = np.convolve(f, f, mode='full')
    max_conv = np.max(conv)
    
    if max_conv < 1e-15:
        return f, max_conv / (np.sum(f)**2) if np.sum(f)>1e-15 else 1e10
    
    # Identify peak indices
    threshold = peak_threshold * max_conv
    peak_indices = np.where(conv >= threshold)[0]
    
    # Compute sensitivity for each bin j
    # d(conv[k])/d(f[j]) = f[k-j] + f[j-k] (with boundary checks)
    # We sum this over all peak indices k
    sensitivity = np.zeros(N)
    
    # Optimization: Instead of iterating over all peaks and all j, 
    # we can use the fact that sensitivity is a convolution-like operation.
    # Let P be a vector where P[k] = 1 if k is a peak index, else 0.
    # Then sensitivity[j] = sum_k P[k] * (f[k-j] + f[j-k])
    # This is essentially 2 * convolve(P, f) but with specific index handling.
    # Let's stick to the direct loop for clarity, as N=600 is small.
    # However, even N^2 is 360k ops. If we have many peaks, it's fine.
    
    # More efficient: 
    # sensitivity[j] = sum_{k in peaks} f[k-j] + f[j-k]
    # Let's create a mask for peaks
    peak_mask = np.zeros(len(conv))
    peak_mask[peak_indices] = 1.0
    
    # term1[j] = sum_k peak_mask[k] * f[k-j]
    # This is cross-correlation of f and peak_mask?
    # conv(f, peak_mask)[j] = sum_i f[i] peak_mask[j-i]
    # We want sum_k peak_mask[k] f[k-j]. Let k = i + j.
    # sum_i peak_mask[i+j] f[i].
    # This is cross-correlation.
    
    # Using FFT for speed if N was larger, but for N=600, direct is ok.
    # Let's use a simple loop over peaks, which is usually < N.
    for k in peak_indices:
        # Contribution to sensitivity[j] from f[k-j]
        # j ranges such that k-j is in [0, N-1] => j in [k-(N-1), k]
        j_min = max(0, k - (N - 1))
        j_max = min(N - 1, k)
        # f[k-j] for j in [j_min, j_max]
        # Indices in f: k-j_min ... k-j_max (reversed)
        # f_idx = k - j
        # If j=j_min, f_idx = k - (k-N+1) = N-1
        # If j=j_max, f_idx = k - k = 0
        # So f[k-j_min : k-j_max+1] is f[N-1 : -1 : -1] roughly
        # Let's just use numpy slicing
        idx_start = k - j_max
        idx_end = k - j_min + 1
        # f[idx_start : idx_end] corresponds to f[k-j_max] ... f[k-j_min]
        # We need to add these to sensitivity[j_max] ... sensitivity[j_min]
        # So we reverse the slice
        vals = f[idx_start:idx_end][::-1]
        sensitivity[j_min:j_max+1] += vals
        
        # Contribution from f[j-k]
        # j-k in [0, N-1] => j in [k, k+N-1]
        j_min2 = max(0, k)
        j_max2 = min(N - 1, k + N - 1)
        if j_min2 <= j_max2:
            idx_start2 = j_min2 - k
            idx_end2 = j_max2 - k + 1
            vals2 = f[idx_start2:idx_end2]
            sensitivity[j_min2:j_max2+1] += vals2
            
    # Normalize sensitivity to have mean 0 and std 1 (or similar)
    # We want to reduce bins with high sensitivity.
    sens_mean = np.mean(sensitivity)
    sens_std = np.std(sensitivity) + 1e-12
    sens_norm = (sensitivity - sens_mean) / sens_std
    
    # Update
    # f_new = f * (1 - alpha * sens_norm)
    # Or f_new = f - alpha * sens_norm * f?
    # Multiplicative update is better for non-negativity and scale invariance.
    f_new = f * (1.0 - alpha * sens_norm)
    f_new = np.maximum(f_new, 0.0)
    
    # Rescale to maintain sum = N
    s_new = np.sum(f_new)
    if s_new > 1e-10:
        f_new = f_new * (N / s_new)
    else:
        f_new = np.ones(N)
        
    return f_new, compute_score(f_new)

def construct():
    start_time = time.time()
    
    # 1. Initialization
    # Try a few different cosine initializations and pick the best
    rng = np.random.RandomState(42)
    best_f = None
    best_score = np.inf
    
    n_components_options = [2, 3, 4, 5]
    for n_c in n_components_options:
        f_init = generate_cosine_initial(n_c, rng)
        score = compute_score(f_init)
        if score < best_score:
            best_score = score
            best_f = f_init.copy()
            
    # If all failed, fallback to constant
    if best_f is None:
        best_f = np.ones(N)
        best_score = compute_score(best_f)
        
    current_f = best_f.copy()
    
    # 2. Main Peak Shaving Loop
    # Start with larger alpha, decrease over time
    alpha = 0.05
    peak_threshold = 0.95
    
    while time.time() - start_time < TIME_LIMIT - 10.0:
        f_new, new_score = peak_shaving_step(current_f, alpha, peak_threshold)
        
        if new_score < best_score:
            best_score = new_score
            best_f = f_new.copy()
            current_f = f_new.copy()
            # Slightly increase alpha if improving
            alpha = min(alpha * 1.01, 0.15)
        else:
            # Decrease alpha if not improving
            alpha *= 0.95
            if alpha < 1e-4:
                break
        
        # Tighten peak threshold slightly over time to focus on the very top
        # This helps in the final polishing stage
        if alpha < 0.02:
            peak_threshold = min(peak_threshold + 0.001, 0.99)

    # 3. Final Polish Pass
    # Use very small alpha and high threshold
    alpha = 0.005
    peak_threshold = 0.99
    
    polish_start = time.time()
    polish_duration = min(5.0, TIME_LIMIT - (time.time() - start_time) - 2.0)
    
    if polish_duration > 0:
        current_f = best_f.copy()
        while time.time() - polish_start < polish_duration:
            f_new, new_score = peak_shaving_step(current_f, alpha, peak_threshold)
            if new_score < best_score:
                best_score = new_score
                best_f = f_new.copy()
                current_f = f_new.copy()
            else:
                alpha *= 0.9
                if alpha < 1e-6:
                    break

    return list(best_f)

# EVOLVE-BLOCK-END
