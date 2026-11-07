import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Cost-Adjusted Lower Confidence Bound (LCB).
    
    Args:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP standard deviations
        f_best: float - best observed value so far
        cost: array of shape (2000,) - evaluation costs
    
    Returns:
        array of shape (2000,) - acquisition scores (higher = better)
    """
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-10)
    
    # Lower Confidence Bound for minimization
    # We want to minimize, so we prefer low mu and high sigma (uncertainty).
    # LCB = mu - kappa * sigma
    # To maximize the score, we negate the LCB.
    kappa = 2.0
    lcb = mu - kappa * sigma_safe
    
    # Cost-aware score: higher is better
    # We negate the LCB so that lower function values (better) yield higher scores.
    # Dividing by cost penalizes expensive evaluations.
    score = -lcb / cost_safe
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
