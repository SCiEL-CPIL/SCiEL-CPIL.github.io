import numpy as np
import time
from scipy.stats import norm

_start_time = time.time()
_DEADLINE = 325.0  # 5s margin

def acquisition(mu, sigma, f_best):
    # Check deadline
    if time.time() > _DEADLINE:
        # Fallback: return the point with lowest mu (exploitation)
        # This is a safe fallback
        return -mu
        
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    
    # Ensure sigma is positive to avoid division issues, though it should be
    sigma_safe = np.maximum(sigma, 1e-9)
    
    z = (f_best - mu) / sigma_safe
    
    # Compute EI
    scores = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
