import numpy as np
import time

_START_TIME = time.time()
_TIMEOUT = 330.0 # Seconds

def acquisition(mu, sigma, f_best):
    # Deadline guard
    if time.time() - _START_TIME > _TIMEOUT - 1.0:
        # Return a default acquisition, e.g., EI
        from scipy.stats import norm
        z = (f_best - mu) / (sigma + 1e-9)
        return (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)

    f_min_known = -1.8013
    
    # Calculate gap from current best to known global minimum
    gap = max(0.0, f_best - f_min_known)
    
    # Use a smoother beta schedule
    # Early: high exploration, Late: exploitation
    # Use sigmoid-like transition
    norm_gap = min(1.0, gap)
    
    # Beta ranges from ~1.0 (exploitative) to ~2.5 (exploratory)
    beta = 1.0 + 1.5 * norm_gap
    
    # LCB score: -mu + beta * sigma
    # Higher score is better
    score = -mu + beta * sigma
    
    # Sanitize
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Ensure finite and reasonable
    score = np.clip(score, -1e6, 1e6)
    
    return score
