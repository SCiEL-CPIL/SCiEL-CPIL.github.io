"""Acquisition function for Bayesian optimization of branin 2d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE_MARGIN = 5.0  # seconds before the 330s limit


def _check_deadline():
    """Return True if we should stop to avoid timeout."""
    return (time.time() - _start_time) > (330.0 - _DEADLINE_MARGIN)


def acquisition(mu, sigma, f_best):
    """
    Acquisition function: Probability of Improvement (PoI) for Branin-2D.
    
    PoI scores candidates by P(f(x) < f_best) = Phi((f_best - mu) / sigma).
    This decouples exploration from the magnitude of improvement, favoring
    broad coverage of the multi-modal Branin landscape over EI's tendency
    to focus on high-improvement regions.
    
    For minimization, we want P(f(x) < f_best), which is:
        Phi((f_best - mu) / sigma)
    
    Parameters:
        mu: (2000,) array of GP means
        sigma: (2000,) array of GP std devs
        f_best: lowest observed function value
    
    Returns:
        (2000,) array of scores, higher = more desirable
    """
    # Deadline guard: if we're running out of time, return a safe fallback
    if _check_deadline():
        sigma_safe = np.maximum(sigma, 1e-8)
        z = (f_best - mu) / sigma_safe
        poi = norm.cdf(z)
        return np.where(np.isfinite(poi), poi, 0.0)
    
    # Clip sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # z-score: (f_best - mu) / sigma
    # For minimization, we want P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement: higher = more likely to beat f_best
    poi = norm.cdf(z)
    
    # Ensure finite scores
    score = np.where(np.isfinite(poi), poi, 0.0)
    
    return score
# EVOLVE-BLOCK-END
