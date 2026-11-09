# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_START_TIME = time.time()
_DEADLINE = _START_TIME + 290.0  # 300s limit, 10s safety margin

def acquisition(mu, sigma, f_best):
    """
    Expected Improvement (EI) with exploration bonus for minimization.
    
    Combines EI with a scaled uncertainty term to encourage exploration.
    
    Parameters:
    mu: array of shape (N,)
    sigma: array of shape (N,)
    f_best: scalar, lowest observed value
    
    Returns:
    scores: array of shape (N,), higher values indicate better candidates
    """
    # Check deadline
    if time.time() > _DEADLINE:
        # Fallback: simple exploitation
        return -mu
    
    # Avoid division by zero or invalid values
    sigma_safe = np.where(sigma > 1e-10, sigma, 1e-10)
    
    # Standardized improvement: (f_best - mu) / sigma
    # For minimization, improvement is when mu < f_best
    Z = (f_best - mu) / sigma_safe
    
    # Expected Improvement formula:
    # EI = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    # where Phi is the standard normal CDF and phi is the standard normal PDF
    # Note: Z = (f_best - mu) / sigma, so when mu < f_best, Z > 0
    EI = (f_best - mu) * norm.cdf(Z) + sigma_safe * norm.pdf(Z)
    
    # Add a small exploration bonus to encourage visiting uncertain regions
    # This helps escape local optima in the RBF mixture landscape
    exploration_bonus = 0.1 * sigma_safe
    
    scores = EI + exploration_bonus
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
