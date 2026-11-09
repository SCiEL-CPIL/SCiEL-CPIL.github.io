import numpy as np
import time
from scipy.stats import norm

# Global start time for deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 320.0  # 320s limit to leave margin

def acquisition(mu, sigma, f_best):
    # Deadline guard
    if time.time() > _DEADLINE:
        return np.zeros_like(mu)

    # Expected Improvement for minimization
    # EI = E[min(0, f_best - f(x))]
    # For a Gaussian posterior, this is:
    # (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    # Note: We want to MAXIMIZE the acquisition score.
    # Standard EI for minimization is positive when mu < f_best.
    
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    scores = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
