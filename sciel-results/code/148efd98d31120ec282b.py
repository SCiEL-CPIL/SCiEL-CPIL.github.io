import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE_MARGIN = 10.0

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of the Goldstein-Price 2D function.
    
    Uses Normalized Expected Improvement (NEI) which has shown strong performance
    for this problem. NEI = EI / sigma, which naturally balances exploration and
    exploitation by normalizing the expected improvement by the uncertainty.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean of the GP posterior at 2000 candidate points
    sigma : ndarray, shape (2000,)
        Standard deviation of the GP posterior at 2000 candidate points
    f_best : float
        Best observed value so far (lowest value)
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means more promising to evaluate next
    """
    # Deadline guard
    current_time = time.time()
    remaining_time = 330.0 - (current_time - _START_TIME)
    
    if remaining_time < _DEADLINE_MARGIN:
        # Fallback: Greedy (lowest mu)
        scores = -mu
        return np.where(np.isfinite(scores), scores, -np.inf)
    
    # Safe sigma to avoid division by zero
    eps = 1e-9
    sigma_safe = sigma + eps
    
    # Calculate z-score for EI
    z = (f_best - mu) / sigma_safe
    
    # Calculate Expected Improvement (EI) for minimization
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Normalized EI: EI / sigma
    # This is the key structural choice: normalizing by sigma makes the score
    # robust to the scale of uncertainty and naturally balances exploration
    # (high sigma regions) with exploitation (high EI regions)
    scores = ei / sigma_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
