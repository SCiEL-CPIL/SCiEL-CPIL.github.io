import time
import numpy as np

N = 600
TOTAL_TIME = 150.0
SAFETY_MARGIN = 5.0

def _autoconv_peak(f):
    """Compute the normalized maximum of the autoconvolution of f."""
    dx = 1.0 / (2.0 * N)
    size = 2 * N
    f_padded = np.zeros(size)
    f_padded[:N] = f
    
    F = np.fft.fft(f_padded, n=2*size)
    conv = np.fft.ifft(F * F).real[:size]
    
    peak = np.max(conv) * dx
    l1 = np.sum(f) * dx
    normalized_peak = peak / (l1 * l1)
    
    return normalized_peak

def _compute_conv(f):
    """Compute the autoconvolution of f on the full grid."""
    dx = 1.0 / (2.0 * N)
    size = 2 * N
    f_padded = np.zeros(size)
    f_padded[:N] = f
    
    F = np.fft.fft(f_padded, n=2*size)
    conv = np.fft.ifft(F * F).real[:size]
    
    return conv, dx

def _waterfilling_iteration(f, conv, dx, target_peak, step_size):
    """
    Perform one water-filling iteration.
    
    The idea: if the autoconvolution peak at some shift k is above target_peak,
    we need to reduce the contributions to that peak. The autoconvolution at shift k
    is conv[k] = sum_j f[j] * f[k-j]. To reduce this, we can decrease f at positions
    that contribute most to the peak.
    
    Conversely, if the peak is below target, we can increase f where it helps fill
    the valleys.
    
    This is inspired by water-filling in information theory: we distribute "water"
    (signal energy) to equalize the water level (autoconvolution profile).
    """
    f_new = f.copy()
    size = len(conv)
    
    # Find the current peak and its location
    peak_val = np.max(conv) * dx
    peak_idx = np.argmax(conv)
    
    # The normalized peak
    l1 = np.sum(f) * dx
    norm_peak = peak_val / (l1 * l1)
    
    # For water-filling, we want to flatten the convolution profile.
    # Strategy: compute the "sensitivity" of each f[i] to the peak.
    # The peak is at shift k=peak_idx. The contribution of f[i] to conv[peak_idx]
    # comes from pairs (i, peak_idx - i) and (peak_idx - i, i).
    
    # Actually, let's think differently. The autoconvolution at shift k is:
    # conv[k] = sum_{j} f[j] * f[k-j] (discrete approximation)
    # 
    # To reduce the peak, we should decrease f at positions that are "hot" 
    # (contribute to many high-convolution shifts).
    # To increase the minimum, we should increase f at "cold" positions.
    
    # Compute the "heat map": for each position i in f, how much does it contribute
    # to the overall convolution energy?
    # heat[i] = sum_k conv[k] * (contribution of f[i] to conv[k])
    # 
    # A simpler approach: use the gradient of the peak with respect to f.
    
    # Compute gradient of conv[peak_idx] with respect to f
    grad = np.zeros(N)
    k = peak_idx
    for i in range(N):
        # f[i] contributes to conv[k] through f[i]*f[k-i] and f[k-i]*f[i]
        # If k-i is in [0, N), then f[i] appears in the product f[i]*f[k-i]
        # and f[k-i]*f[i] (same term, so factor of 2 if i != k-i)
        j = k - i
        if 0 <= j < N:
            if i == j:
                # f[i]*f[i] = f[i]^2, derivative is 2*f[i]
                grad[i] += 2.0 * f[i]
            else:
                # f[i]*f[j] + f[j]*f[i] = 2*f[i]*f[j], derivative w.r.t f[i] is 2*f[j]
                grad[i] += 2.0 * f[j]
    
    # Normalize gradient
    grad_norm = np.linalg.norm(grad)
    if grad_norm > 1e-15:
        grad = grad / grad_norm
    
    # Water-filling: decrease f where gradient is high (reduces peak),
    # increase f where gradient is low (fills valleys)
    # Use a smoothed version
    smooth_grad = np.convolve(grad, np.ones(5)/5.0, mode='same')
    
    # Update: move f in the direction that flattens the convolution
    # f_new[i] = f[i] - step_size * (smooth_grad[i] - mean(smooth_grad))
    grad_mean = np.mean(smooth_grad)
    delta = step_size * (smooth_grad - grad_mean)
    
    f_new = f - delta
    f_new = np.maximum(f_new, 0.0)
    
    # Preserve L1 norm approximately
    l1_old = np.sum(f)
    l1_new = np.sum(f_new)
    if l1_new > 1e-15:
        f_new *= l1_old / l1_new
    
    return f_new, norm_peak

def _waterfilling_optimize(f_init, max_iter, step_size_init, deadline):
    """
    Run the water-filling optimization.
    """
    f = f_init.copy()
    best_f = f.copy()
    best_val = _autoconv_peak(f)
    
    step_size = step_size_init
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        conv, dx = _compute_conv(f)
        f_new, norm_peak = _waterfilling_iteration(f, conv, dx, best_val, step_size)
        
        val = _autoconv_peak(f_new)
        
        if val < best_val:
            best_val = val
            best_f = f_new.copy()
            # Increase step size slightly
            step_size *= 1.05
        else:
            # Decrease step size
            step_size *= 0.9
            # If step size is too small, break
            if step_size < 1e-8:
                break
        
        f = f_new
    
    return best_f, best_val

def construct():
    start_time = time.time()
    deadline = start_time + TOTAL_TIME - SAFETY_MARGIN
    
    t = np.linspace(-0.25, 0.25, N, endpoint=False)
    
    # Try several initializations
    initializations = []
    
    # Uniform
    initializations.append(np.ones(N))
    
    # Cosine bump
    initializations.append(0.5 + 0.5 * np.cos(np.pi * t))
    
    # Gaussian
    initializations.append(np.exp(-20 * t**2))
    
    # Two-bump
    init2 = np.ones(N)
    init2[:N//4] *= 1.5
    init2[3*N//4:] *= 0.5
    initializations.append(init2)
    
    # Evaluate all and pick best
    best_f = None
    best_val = float('inf')
    
    for init in initializations:
        init = np.maximum(0, init)
        val = _autoconv_peak(init)
        if val < best_val:
            best_val = val
            best_f = init.copy()
    
    # Run water-filling optimization
    if time.time() < deadline - 10:
        f_opt, val_opt = _waterfilling_optimize(
            best_f, 
            max_iter=200, 
            step_size_init=0.1,
            deadline=deadline - 5
        )
        if val_opt < best_val:
            best_val = val_opt
            best_f = f_opt.copy()
    
    # Final polish with coordinate descent
    if time.time() < deadline - 5:
        f = best_f.copy()
        current_val = best_val
        
        for _ in range(30):
            if time.time() > deadline - 5:
                break
            improved = False
            for idx in range(N):
                if time.time() > deadline - 5:
                    break
                old_val = f[idx]
                for delta in [-0.003, 0.003]:
                    new_val = max(0.0, old_val + delta)
                    f[idx] = new_val
                    new_score = _autoconv_peak(f)
                    if new_score < current_val:
                        current_val = new_score
                        improved = True
                        if current_val < best_val:
                            best_val = current_val
                            best_f = f.copy()
                    else:
                        f[idx] = old_val
    
    return list(best_f)

if __name__ == "__main__":
    result = construct()
    print(f"Result: {len(result)} values, sum={sum(result):.4f}")
