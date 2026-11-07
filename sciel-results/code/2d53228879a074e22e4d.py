import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_TIMEOUT = 290  # 290 seconds to leave safety margin before 300s limit

def acquisition(mu, sigma, f_best, cost):
    # Check if we're close to timeout - return simple score if so
    if time.time() - _START_TIME > _TIMEOUT:
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        z = np.clip(z, -10, 10)
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.maximum(ei, 0)
        return ei / cost
    
    # Ensure sigma is non-negative and not zero
    sigma = np.maximum(sigma, 1e-9)
    
    # Calculate z-score
    z = (f_best - mu) / sigma
    
    # Clip z to avoid numerical underflow/overflow in pdf/cdf
    z = np.clip(z, -10, 10)
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure EI is non-negative (numerical safety)
    ei = np.maximum(ei, 0)
    
    # Cost-aware score with slight exploration bonus for high-uncertainty regions
    # The cost function is 1 + mean(x scaled to [0,1]), so cost is in [1, 2]
    # We use a slightly more aggressive penalty to favor cheaper evaluations
    score = ei / (cost ** 1.05)
    
    return score
