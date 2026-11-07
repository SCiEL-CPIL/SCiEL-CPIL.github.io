import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 325.0

def acquisition(mu, sigma, f_best):
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        return -mu

    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Adaptive Beta based on uncertainty levels
    # High uncertainty (early) -> High Beta (Explore)
    # Low uncertainty (late) -> Low Beta (Exploit)
    min_sigma = np.min(sigma_safe)
    med_sigma = np.median(sigma_safe)
    
    # Ratio indicates how "converged" the GP is in the best region
    # If min_sigma is much smaller than med_sigma, we have found a peak/valley
    ratio = min_sigma / (med_sigma + 1e-9)
    
    # Beta ranges from 0.5 (exploit) to 3.0 (explore)
    # If ratio is small (converged), beta should be small.
    # If ratio is large (exploring), beta should be large.
    # Let's map ratio [0, 1] to beta [0.5, 3.0]
    # ratio = 1 means min ~ med (no convergence yet)
    # ratio = 0 means min << med (converged)
    beta = 0.5 + 2.5 * ratio
    
    # UCB for minimization: -mu + beta * sigma
    score = -mu + beta * sigma_safe
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
