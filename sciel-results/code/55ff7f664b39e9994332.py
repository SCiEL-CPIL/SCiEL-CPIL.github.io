import numpy as np
from scipy.stats import norm
import time

# Global state for deadline guard
_start_time = time.time()
_TIME_LIMIT = 320.0  # Safety margin from 330s

def _check_deadline():
    """Check if we are approaching the time limit."""
    return (time.time() - _start_time) > _TIME_LIMIT

def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling proxy acquisition function.
    
    Uses normalized improvement heuristic: (f_best - mu) / (sigma + epsilon)
    This encourages sampling points with high potential relative to their uncertainty.
    
    Args:
        mu: Mean of the GP posterior, shape (2000,)
        sigma: Standard deviation of the GP posterior, shape (2000,)
        f_best: Best observed function value so far.
        
    Returns:
        scores: Array of shape (2000,) where higher scores are preferred.
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    
    # Handle edge cases for numerical stability
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # Thompson Sampling proxy: normalized improvement
    # Higher score means higher potential improvement relative to uncertainty
    scores = (f_best - mu) / sigma_safe
    
    # Deadline Guard
    if _check_deadline():
        # Return the already-computed scores
        return np.where(np.isfinite(scores), scores, 0.0)
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
