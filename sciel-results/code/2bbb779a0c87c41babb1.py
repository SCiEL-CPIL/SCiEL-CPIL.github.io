import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 280.0

def acquisition(mu, sigma, f_best):
    # Deadline guard
    if time.time() - _START_TIME > _TIME_LIMIT:
        return -mu
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    
    # Handle edge case where f_best is very small
    # If f_best is 0, we are done, but let's be safe.
    if f_best <= 1e-9:
        # Pure exploitation
        return -mu
    
    # Use raw sigma for UCB
    # This preserves the magnitude of uncertainty
    # We use a fixed exploration coefficient, which is a standard UCB approach
    # If f_best is very large, we might want more exploration, but raw sigma
    # is naturally large in unexplored regions.
    beta = 1.0
    
    # UCB Heuristic
    # Score = -mu + beta * sigma
    # Higher score is better
    scores = -mu + beta * sigma
    
    # Ensure finite scores
    # If any NaNs or Infs appear, replace with a very low score
    scores = np.nan_to_num(scores, nan=-1e9, posinf=1e9, neginf=-1e9)
    
    return scores
