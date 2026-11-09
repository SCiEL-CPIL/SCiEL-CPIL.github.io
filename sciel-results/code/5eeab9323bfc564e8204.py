import numpy as np
from scipy.stats import norm
import time

_start_time = None

def acquisition(mu, sigma, f_best, cost):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: if we are running out of time, use a simpler heuristic
    # to ensure we return quickly and don't contribute to a timeout.
    elapsed = time.time() - _start_time
    if elapsed > 320.0:
        # Simple exploitation score
        return -mu / (cost + 1e-9)

    # Calculate z-score
    # Ensure sigma is not zero
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_z + sigma * pdf_z
    
    # Cost-aware score
    # Using square root of cost to reduce the penalty for higher costs
    # compared to the linear penalty in the seed.
    score = ei / np.sqrt(cost + 1e-9)
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
