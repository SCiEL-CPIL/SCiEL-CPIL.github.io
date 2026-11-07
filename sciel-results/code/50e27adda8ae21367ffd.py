import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Michalewicz 2D.
    Uses Expected Improvement (EI) with robust numerical handling.
    """
    # Explicit deadline guard as per directive, though EI is fast.
    # If this function hangs, return safe scores.
    start_time = time.time()
    DEADLINE = 295.0  # Safety margin
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive and handle numerical stability
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Compute Expected Improvement (for minimization)
    # z = (f_best - mu) / sigma
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    
    # Check time before expensive operations (though these are vectorized C-ops)
    if time.time() - start_time > DEADLINE:
        return np.zeros_like(mu)
        
    z = (f_best - mu) / sigma_safe
    
    # norm.cdf and norm.pdf are stable for typical ranges.
    # Clip z to avoid potential overflow in pdf for extremely negative z?
    # pdf(z) is ~0 for z < -30. cdf(z) is ~0 for z < -30.
    # For z > 30, cdf is 1, pdf is 0.
    # No clipping needed for standard double precision in this range.
    
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure all scores are finite
    ei = np.nan_to_num(ei, nan=0.0, posinf=0.0, neginf=0.0)
    
    return ei
