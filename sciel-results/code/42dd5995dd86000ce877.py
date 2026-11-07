# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Track elapsed time for deadline guard
_start_time = time.time()
_DEADLINE = 325.0  # Safety margin before 330s kill

def acquisition(mu, sigma, f_best):
    # Check deadline
    if time.time() - _start_time > _DEADLINE:
        # Fallback: return UCB-like scores quickly
        sigma_safe = np.maximum(sigma, 1e-9)
        return -mu + 1.5 * sigma_safe
    
    n = len(mu)
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute Expected Improvement (EI) for minimization
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Compute UCB for minimization
    # For minimization, we want low mu, so -mu is high when mu is low
    # UCB: -mu + beta * sigma
    
    # Use a well-tuned fixed beta for UCB
    # Beta around 1.5-2.0 works well for exploration in 1D
    beta = 1.8
    
    # UCB score for minimization
    ucb = -mu + beta * sigma_safe
    
    # Combine EI and UCB with a fixed weight
    # EI handles exploitation (points likely to beat f_best)
    # UCB handles exploration (points with high uncertainty)
    
    # Normalize both to have similar scale
    # Use robust normalization: center and scale by IQR
    def robust_normalize(arr):
        q25, q75 = np.percentile(arr, [25, 75])
        iqr = q75 - q25
        if iqr < 1e-12:
            return np.zeros_like(arr)
        median = np.median(arr)
        return (arr - median) / iqr
    
    ei_norm = robust_normalize(ei)
    ucb_norm = robust_normalize(ucb)
    
    # Weight: 60% EI, 40% UCB
    # This balances exploitation and exploration effectively
    score = 0.6 * ei_norm + 0.4 * ucb_norm
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
