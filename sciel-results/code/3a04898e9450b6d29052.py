"""Rewritten program using a different acquisition approach: Cost-Aware UCB (Upper Confidence Bound)."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Upper Confidence Bound (UCB) acquisition function.
    
    Parameters:
    - mu: array of shape (2000,) - GP mean predictions
    - sigma: array of shape (2000,) - GP standard deviation predictions
    - f_best: scalar - best (lowest) observed function value so far
    - cost: array of shape (2000,) - evaluation costs for each candidate
    
    Returns:
    - scores: array of shape (2000,) - acquisition scores (higher = better)
    """
    # Ensure sigma is non-negative
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # UCB for minimization: u(x) = mu(x) - beta * sigma(x)
    # We want to minimize u(x), so we maximize -u(x) = beta * sigma(x) - mu(x)
    # Beta controls the exploration-exploitation trade-off.
    beta = 1.5
    
    ucb_values = beta * sigma_safe - mu
    
    # Cost-aware score
    scores = ucb_values / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
