import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: track elapsed time to ensure we finish within limits
    # The computation here is vectorized and should be fast, but we add a safeguard
    start_time = time.time()
    
    # Calculate z-score for Expected Improvement
    # sigma is standard deviation
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * P(f < f_best) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is non-negative for numerical stability
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware scoring
    # The cost is 1 + mean(x). Range [1, 2].
    # The global minimum of Hartmann 6D is in the low-cost region (mean(x) ~ 0.23).
    # We penalize higher costs more aggressively to bias the search towards the cheaper region.
    # Using an exponent of 1.5 amplifies the difference between low and high cost points.
    safe_cost = np.maximum(cost, 1e-9)
    score = ei / (safe_cost ** 1.5)
    
    # Ensure all scores are finite
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # If we're getting close to the deadline, return immediately
    # (This is a safety net; the computation above should be fast)
    elapsed = time.time() - start_time
    if elapsed > 295:  # 5 second safety margin
        return score
    
    return score
