# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = None
_TOTAL_BUDGET = 330.0
_SAFETY_MARGIN = 5.0

def _get_elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME

def _deadline_check(safety_margin=_SAFETY_MARGIN):
    """Return True if we are safe to continue."""
    return _get_elapsed() < _TOTAL_BUDGET - safety_margin

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of beale 2d.
    Uses Expected Improvement with an adaptive exploration term that
    increases as we approach the time deadline, encouraging broader
    exploration in the early-to-mid phase and tighter exploitation
    near the end.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Safety: ensure finite inputs
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    
    # Avoid division by zero in sigma
    sigma_safe = np.where(sigma > 1e-12, sigma, 1e-12)
    
    # Compute Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z) where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is non-negative (it should be, but guard against numerical issues)
    ei = np.maximum(ei, 0.0)
    
    # Adaptive exploration: add a term proportional to sigma that
    # scales with remaining time fraction
    elapsed = _get_elapsed()
    remaining_frac = max(0.0, 1.0 - elapsed / _TOTAL_BUDGET)
    
    # Exploration weight: higher when more time remains
    # At start (remaining_frac ~ 1.0): strong exploration
    # Near end (remaining_frac ~ 0.0): pure exploitation
    exploration_weight = 0.5 * remaining_frac
    
    # Add exploration bonus proportional to uncertainty
    exploration_bonus = exploration_weight * sigma_safe
    
    # Combine: EI + exploration bonus
    score = ei + exploration_bonus
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
