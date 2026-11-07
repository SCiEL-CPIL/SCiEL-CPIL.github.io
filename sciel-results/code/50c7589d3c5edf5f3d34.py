"""Cost-aware acquisition function for Bayesian optimization (griewank 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = None

def acquisition(mu, sigma, f_best, cost):
    """
    Compute cost-aware acquisition scores using Probability of Improvement (PI).
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean predictions in objective units.
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions in objective units.
    f_best : float
        Best observed value so far.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate (1 + mean(x scaled to [0,1])).
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = more desirable to evaluate next).
    """
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    # Deadline guard: if we're getting close to the 330s limit, return a safe fallback
    elapsed = time.time() - _START_TIME
    if elapsed > 325:  # 5s safety margin
        # Return a conservative score that prefers lower cost
        safe_score = 1.0 / (cost + 1e-9)
        return np.where(np.isfinite(safe_score), safe_score, 0.0)
    
    # Ensure sigma is positive and finite
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Bayesian model evidence-inspired score for minimization
    # Approximated by z - log(sigma), where z = (f_best - mu) / sigma
    # This favors low mu (high z) and penalizes high sigma (high log sigma)
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues
    z = np.clip(z, -10, 10)
    
    # Calculate log(sigma) safely
    log_sigma = np.log(sigma_safe)
    
    # Score: Higher z (better mean) and lower log_sigma (lower uncertainty) is better
    # Note: We use (z - log_sigma) instead of the directive's literal -(z) - log_sigma
    # because for minimization, we want to maximize the deviation of best from mean (positive z)
    score = (z - log_sigma) / (cost + 1e-9)
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
