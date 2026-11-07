import numpy as np
from scipy.stats import norm
import time

# Global timer for safety
_START_TIME = time.time()
_TIME_LIMIT = 300.0  # Safety margin below 330s

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Cost-Aware EI.
    Returns scores of shape (2000,), where higher scores indicate better candidates.
    """
    # Ensure safety in denominators
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Expected Improvement (EI) for minimization
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost-Aware EI
    scores = ei / cost_safe
    
    # Ensure all outputs are finite
    scores = np.where(np.isfinite(scores), scores, -1e9)
    
    return scores
