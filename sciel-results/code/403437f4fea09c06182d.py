"""Rewritten program using a Cost-aware Lower Confidence Bound (LCB) strategy."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    """
    Acquisition function for cost-aware Bayesian optimization.
    
    Uses Cost-aware Lower Confidence Bound (LCB) for minimization.
    LCB encourages exploration by adding a term proportional to the uncertainty (sigma).
    For minimization, we want to minimize (mu - kappa * sigma).
    We return (kappa * sigma - mu) to make higher scores better.
    Dividing by cost prioritizes evaluations that offer high promise per unit cost.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean predictions of the GP
    sigma : ndarray, shape (2000,)
        Standard deviation predictions of the GP
    f_best : float
        Best observed value so far (minimum)
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next)
    """
    # Ensure sigma is positive to avoid issues
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-6)
    
    # Exploration-exploitation trade-off parameter
    # A value of 2.5 is a heuristic estimate for ~20 iterations
    kappa = 2.5
    
    # LCB for minimization: L(x) = mu(x) - kappa * sigma(x)
    # We want to minimize L(x), so we maximize -L(x) = kappa * sigma(x) - mu(x)
    lcb_score = kappa * sigma_safe - mu
    
    # Cost-aware score: normalize by the evaluation cost
    score = lcb_score / cost_safe
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
