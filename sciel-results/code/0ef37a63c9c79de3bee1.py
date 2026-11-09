import numpy as np
import time

# Global start time for deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 320.0  # 320s limit to leave margin

def acquisition(mu, sigma, f_best):
    # Deadline guard
    if time.time() > _DEADLINE:
        # Return a neutral score if we are out of time, 
        # though this should rarely happen for vectorized ops.
        # Returning zeros might be safer than crashing.
        return np.zeros_like(mu)

    # UCB-like acquisition for minimization
    # We want to maximize the score, where higher score means more promising.
    # For minimization, we look at the lower confidence bound: mu - beta * sigma
    # To maximize this, we return -mu + beta * sigma
    
    beta = 2.5
    scores = -mu + beta * sigma
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
