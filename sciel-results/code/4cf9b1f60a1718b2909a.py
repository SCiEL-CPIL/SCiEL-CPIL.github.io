"""Rewritten program using Cost-Weighted Upper Confidence Bound (cUCB) for minimization."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    """
    Acquisition function for cost-aware Bayesian optimization using cUCB.
    
    This approach uses an Upper Confidence Bound strategy adapted for minimization.
    Instead of maximizing expected improvement, we maximize a score that balances
    the lower bound of the function value (optimism in the face of uncertainty)
    and the cost of evaluation.
    
    For minimization, we want to find points with low predicted mean and high uncertainty.
    The UCB for minimization can be viewed as maximizing:
        Score_base = beta * sigma - mu
    
    We then normalize this by the cost of evaluation to prefer cheaper points.
    
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
    # Ensure sigma is non-negative
    sigma_safe = np.maximum(sigma, 0.0)
    
    # Cost safety
    cost_safe = np.maximum(cost, 1e-6)
    
    # UCB exploration parameter
    # A fixed beta is used. We can tune this, but 1.0-2.0 is common.
    # Since we are minimizing, a higher beta encourages more exploration.
    beta = 2.0
    
    # Base score for minimization: high sigma and low mu are good.
    # We want to maximize (beta * sigma - mu).
    # Note: mu is the predicted value. Lower mu is better.
    # So -mu is a "goodness" term.
    base_score = beta * sigma_safe - mu
    
    # Cost-aware weighting
    # Divide by cost to prefer cheaper evaluations.
    # Using cost^1.5 to strongly penalize expensive points.
    score = base_score / (cost_safe ** 1.5)
    
    # Ensure scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Optional: Shift scores so that the minimum is 0?
    # Not necessary for selection, but ensures non-negative scores if desired.
    # The task says "higher = evaluate next", so relative values matter.
    
    return score
# EVOLVE-BLOCK-END
