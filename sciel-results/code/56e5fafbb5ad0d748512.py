import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE = 320.0  # Safety margin

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard
    if time.time() > _DEADLINE:
        # Return a simple, fast score to avoid timeout
        # Just return 1/cost to be cost-aware but simple
        return 1.0 / (cost + 1e-9)
    
    # Avoid division by zero
    sigma_safe = sigma + 1e-9
    
    # Probability of Improvement
    z = (f_best - mu) / sigma_safe
    pi = norm.cdf(z)
    
    # Exploration term: normalized sigma
    # Using sigma / (sigma + 1) keeps it in [0, 1)
    exploration = sigma / (sigma + 1.0)
    
    # Combine
    beta = 0.2
    score = (pi + beta * exploration) / (cost + 1e-9)
    
    # Ensure finite
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
