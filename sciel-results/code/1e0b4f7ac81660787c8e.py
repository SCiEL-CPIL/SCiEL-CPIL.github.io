import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard
    if time.time() - _start_time > 320:
        return 1.0 / cost

    # Compute z-score
    # Add small epsilon to sigma to avoid division by zero
    eps = 1e-9
    z = (f_best - mu) / (sigma + eps)
    
    # Clip z to prevent numerical issues in cdf/pdf
    z = np.clip(z, -10, 10)
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Cost-aware score
    score = ei / cost
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
