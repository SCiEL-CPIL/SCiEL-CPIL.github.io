"""Cost-aware acquisition for rastrigin 2d BO loop."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

# Deadline guard: the evaluator kills after 330s wall-clock.
_T0 = time.time()
_DEADLINE = 325.0  # safety margin of 5s before the 330s kill


def _deadline_exceeded():
    return (time.time() - _T0) > _DEADLINE


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Probability of Improvement (PI)
    with a bounded Uncertainty Bonus.
    
    This structural change replaces the LCB/EI blend with a PI-based approach.
    PI is robust for multimodal functions like Rastrigin as it directly measures
    the likelihood of beating the current best. The Uncertainty Bonus provides
    a smooth, bounded exploration signal that is less aggressive than LCB.
    
    Parameters:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP standard deviation predictions
        f_best: float - best observed value so far
        cost: array of shape (2000,) - cost of evaluating each candidate
    
    Returns:
        array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    # Deadline guard: if we're running out of time, return a simple fallback
    if _deadline_exceeded():
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        pi = norm.cdf(z)
        return pi / (cost + 1e-9)
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: ensure finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    
    # Ensure sigma and cost are positive
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Compute z-score
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -10, 10)
    
    # Probability of Improvement
    pi = norm.cdf(z)
    
    # Uncertainty Bonus: bounded by 1, encourages exploration of high-sigma regions
    # This is structurally different from LCB's linear beta*sigma term
    uncertainty_bonus = sigma_safe / (1.0 + sigma_safe)
    
    # Base score is PI / cost
    score = pi / cost_safe
    
    # Add uncertainty bonus scaled by a small weight
    # The weight is adaptive: higher when f_best is large (early exploration)
    if f_best < 1.0:
        weight = 0.1
    elif f_best < 5.0:
        weight = 0.2
    else:
        weight = 0.3
        
    score += weight * uncertainty_bonus / cost_safe
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
