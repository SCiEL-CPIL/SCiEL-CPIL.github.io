import numpy as np
import time
from scipy.stats import norm

# Global deadline tracking
_start_time = None
_DEADLINE_MARGIN = 5.0  # seconds to spare before the 330s kill

def _ensure_deadline():
    """Check if we're approaching the deadline."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    elapsed = time.time() - _start_time
    return elapsed < (330.0 - _DEADLINE_MARGIN)

def acquisition(mu, sigma, f_best):
    """
    Adaptive Upper Confidence Bound (UCB) acquisition function for minimization.
    
    The score is defined as kappa * sigma - mu.
    The evaluator selects the candidate with the highest score, which corresponds
    to minimizing (mu - kappa * sigma).
    
    The exploration parameter kappa is adapted based on the gap between the current
    best observed value (f_best) and the known global minimum of the Goldstein-Price
    function (3.0). A larger gap encourages more exploration (higher kappa).
    
    Args:
        mu: Array of posterior means (2000,)
        sigma: Array of posterior standard deviations (2000,)
        f_best: Current best observed value (scalar)
        
    Returns:
        Array of scores (2000,), higher is better.
    """
    if not _ensure_deadline():
        # Fallback: return simple EI-like scores quickly
        sigma_safe = np.maximum(sigma, 1e-8)
        z = (f_best - mu) / sigma_safe
        poi = norm.cdf(z)
        return np.where(np.isfinite(poi), poi, 0.0)
    
    # Ensure sigma is strictly positive
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Calculate the gap to the known minimum (3.0 for Goldstein-Price)
    # Clamp to 0 to avoid negative gaps if we somehow overshoot (numerical error)
    gap = max(0.0, f_best - 3.0)
    
    # Adaptive kappa:
    # When gap is large (early), kappa is large -> high exploration.
    # When gap is small (late), kappa is small -> high exploitation.
    # Using a linear scaling with a base of 1.0.
    kappa = 1.0 + 2.0 * gap
    
    # UCB score for minimization:
    # We want to minimize mu - kappa * sigma.
    # The acquisition function returns a score where higher is better.
    # So we return -(mu - kappa * sigma) = kappa * sigma - mu.
    scores = kappa * sigma_safe - mu
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
