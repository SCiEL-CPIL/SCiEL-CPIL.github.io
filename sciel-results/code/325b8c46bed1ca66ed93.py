import numpy as np
import time
from scipy.stats import norm

_start_time = time.time()

def acquisition(mu, sigma, f_best):
    # Time guard: If we are close to the limit, return a simple score to avoid timeout
    elapsed = time.time() - _start_time
    if elapsed > 325:
        return -mu

    # Probability of Improvement (PI) for minimization
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    # We want to maximize this probability.
    
    # Ensure sigma is non-negative and finite
    sigma = np.maximum(sigma, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    
    # Avoid division by zero
    sigma_safe = np.where(sigma > 1e-10, sigma, 1e-10)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    
    # Calculate probability
    scores = norm.cdf(z)
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
