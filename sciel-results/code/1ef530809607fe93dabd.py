"""Acquisition function for Bayesian optimization of branin 2d."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


# Deadline guard: keep explicit wall-clock tracking
_START_TIME = time.time()
_TIME_LIMIT = 300.0  # evaluator kills at 330s, leave margin
_TIME_MARGIN = 10.0  # safety margin seconds


def _time_left():
    return _TIME_LIMIT - _TIME_MARGIN - (time.time() - _START_TIME)


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Branin 2D.
    
    Uses a hybrid of Expected Improvement (EI) and Upper Confidence Bound (UCB)
    without min-max normalization to preserve the global structure of the
    acquisition landscape. Branin is multi-modal, so we need a balance between
    exploiting known good regions (EI) and exploring uncertain regions (UCB).
    
    Parameters:
    -----------
    mu : ndarray, shape (n,)
        GP mean at candidate points (objective units)
    sigma : ndarray, shape (n,)
        GP standard deviation at candidate points (objective units)
    f_best : float
        Best observed value so far
    
    Returns:
    --------
    scores : ndarray, shape (n,)
        Acquisition scores (higher = more promising)
    """
    # Check time budget - if we're running out, return a simple fallback
    if _time_left() < 1.0:
        # Fallback: pure EI with small epsilon
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        return (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute Expected Improvement (EI) for minimization
    z = (f_best - mu) / sigma_safe
    # Clip z to avoid numerical issues
    z = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Compute UCB for minimization: UCB = -mu + beta * sigma
    # Higher UCB means more promising (lower predicted value with uncertainty bonus)
    # Use beta=1.0 which performed well in inspiration 1 (score 1.42)
    beta = 1.0
    ucb = -mu + beta * sigma_safe
    
    # Combine EI and UCB with a weighted sum in objective units
    # EI tends to be smaller in magnitude than UCB, so we weight them
    # to balance exploitation and exploration
    # Based on inspiration 3's analysis, a direct weighted sum without
    # normalization preserves the global structure better
    gamma = 0.5  # weight for UCB relative to EI
    score = ei + gamma * ucb
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
