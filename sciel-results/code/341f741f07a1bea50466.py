import numpy as np
import time

N = 400

def _compute_objective(h):
    """Compute max |autoconvolution| / (sum h)^2."""
    # Linear autoconvolution
    conv = np.convolve(h, h, mode='full')
    max_abs = np.max(np.abs(conv))
    sum_h = np.sum(h)
    if sum_h == 0:
        return np.inf
    return max_abs / (sum_h ** 2)

def construct():
    """
    Returns the sequence of signed step heights.
    Uses random search to find a configuration that minimizes
    max |autoconvolution| / (sum h)^2.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    best_h = np.ones(N, dtype=np.float64)
    best_obj = _compute_objective(best_h)
    
    rng = np.random.default_rng(42)
    
    # Try random binary-like vectors with signs
    for trial in range(5000):
        if time.time() > deadline:
            break
        
        # Random signed vector: values in {-1, 0, 1} or continuous
        h = rng.choice([-1.0, 0.0, 1.0], size=N)
        
        # Ensure sum is not zero
        if np.abs(np.sum(h)) < 1e-6:
            continue
        
        obj = _compute_objective(h)
        if obj < best_obj:
            best_obj = obj
            best_h = h.copy()
        
        # Also try smooth random vectors
        if trial % 5 == 0:
            h_smooth = rng.standard_normal(N)
            # Make it have some structure: low-pass filter
            from scipy.ndimage import gaussian_filter1d
            h_smooth = gaussian_filter1d(h_smooth, sigma=2.0)
            # Sign it
            h_signed = np.sign(h_smooth)
            if np.abs(np.sum(h_signed)) < 1e-6:
                continue
            obj = _compute_objective(h_signed)
            if obj < best_obj:
                best_obj = obj
                best_h = h_signed.copy()
    
    return best_h
