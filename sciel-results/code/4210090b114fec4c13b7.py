"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()

def acquisition(mu, sigma, f_best):
    # Deadline guard
    if time.time() - _start_time > 320:
        # Return standard EI as fallback or zeros? 
        # Better to return a valid array quickly.
        z = (f_best - mu) / (sigma + 1e-9)
        return (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)

    # Look-Ahead Expected Improvement (LAEI) Approximation
    # We assume the true global minimum M is likely lower than f_best.
    # We model M as being offset from f_best by a factor of the current uncertainty.
    # This encourages finding points that are significantly better, not just marginally better.
    
    # Use median sigma as a scale for the "look-ahead" distance
    sigma_med = np.median(sigma)
    
    # Adaptive alpha: 
    # Early on (high sigma), look further ahead (larger offset).
    # Late on (low sigma), look closer (smaller offset).
    # Normalize sigma_med by f_best magnitude? Or just use it directly?
    # Hartmann 6d min is -3.32. Range is roughly [-3.3, 3.3].
    # Let's use a fixed alpha scaled by sigma_med.
    alpha = 0.5
    
    f_target = f_best - alpha * sigma_med
    
    # Calculate EI towards f_target
    z = (f_target - mu) / (sigma + 1e-9)
    scores = (f_target - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
