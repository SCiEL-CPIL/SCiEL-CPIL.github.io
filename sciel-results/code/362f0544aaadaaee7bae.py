import numpy as np
import time
from scipy.stats import norm

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = 320.0  # Safety margin before 330s kill


def acquisition(mu, sigma, f_best):
    """
    Quantile Acquisition Function (QAF) based acquisition.
    Maximizes the probability of exceeding a target value (f_best).
    QAF = Phi((f_best - mu) / sigma)
    Higher score means higher probability of finding a new minimum.
    """
    # Deadline check
    if time.time() - _START_TIME > _DEADLINE:
        # Return safe fallback: exploit best mean
        return -mu.copy()
    
    # Ensure inputs are proper arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute the standard normal CDF of (f_best - mu) / sigma
    # This gives the probability that f(x) < f_best, i.e., improvement
    z = (f_best - mu) / sigma_safe
    
    # QAF score: probability of improvement
    # Higher probability = more likely to find a new minimum
    scores = norm.cdf(z)
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
