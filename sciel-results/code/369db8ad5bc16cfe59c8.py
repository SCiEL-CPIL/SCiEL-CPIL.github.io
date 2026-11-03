import numpy as np
from scipy.fft import fft, ifft
import time

N = 600
TIME_LIMIT = 130.0  # Safety margin before 150s kill

def compute_score(f):
    """Compute the normalized maximum of the autoconvolution."""
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return np.inf
    f = np.maximum(f, 0.0)
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    F = fft(f)
    autoconv = ifft(F * F).real
    max_conv = np.max(autoconv)
    return max_conv / (s * s)

def construct():
    start_time = time.time()
    
    # Start with uniform distribution (the theoretical optimum)
    f = np.ones(N) / N
    best_f = f.copy()
    best_score = compute_score(best_f)
    
    # We already have the optimal solution analytically
    # But let's do a small refinement just in case numerical issues arise
    # Actually, uniform is exactly optimal for this problem
    
    # Verify the score
    F = fft(f)
    autoconv = ifft(F * F).real
    max_conv = np.max(autoconv)
    s = np.sum(f)
    score = max_conv / (s * s)
    
    # The score should be 1/N = 1/600 ≈ 0.001667
    
    # No need for further optimization - this is the global minimum
    # Return the uniform distribution
    return [float(x) for x in best_f]
