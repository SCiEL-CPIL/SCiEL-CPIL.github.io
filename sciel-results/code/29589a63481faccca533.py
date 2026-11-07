import numpy as np
from scipy.stats import norm
import time

# Track elapsed time for deadline guard
_start_time = time.time()
TIME_LIMIT = 330.0
SAFETY_MARGIN = 5.0
_eval_count = 0

def acquisition(mu, sigma, f_best):
    global _eval_count
    _eval_count += 1
    
    # Deadline guard: return safe scores if approaching time limit
    elapsed = time.time() - _start_time
    if elapsed > TIME_LIMIT - SAFETY_MARGIN:
        # If we are at the deadline, return a neutral score (e.g., just -mu)
        # to force a selection quickly.
        return -mu
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive and finite
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Thompson Sampling: sample from the posterior and pick the best
    # For minimization, we want the lowest value
    # We can't actually sample and return the index, so we use a deterministic proxy
    # The expected minimum of N samples from N(mu, sigma^2) is approximately:
    # E[min] ≈ mu - sigma * sqrt(2 * log(N)) for large N
    # But since we're scoring 2000 candidates, we can use a simpler approach:
    # score = -mu + sigma * k where k is a small constant
    # This is equivalent to LCB with beta = k
    
    # Use a fixed beta for Thompson Sampling-like behavior
    # Beta = 1.0 corresponds to sampling from N(mu, sigma^2) and picking the min
    # In expectation, this is equivalent to LCB with beta = 1.0
    
    beta = 1.0
    
    # LCB for minimization: -mu + beta * sigma
    scores = -mu + beta * sigma_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
