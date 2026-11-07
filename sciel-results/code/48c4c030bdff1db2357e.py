"""Rewritten program using a different acquisition approach: Upper Confidence Bound (UCB) with cost-awareness."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware UCB acquisition function for Bayesian optimization.
    
    Parameters:
    - mu: array of shape (2000,) - GP mean predictions
    - sigma: array of shape (2000,) - GP standard deviation predictions
    - f_best: scalar - best (lowest) observed function value so far
    - cost: array of shape (2000,) - evaluation costs for each candidate
    
    Returns:
    - scores: array of shape (2000,) - acquisition scores (higher = better)
    """
    # Add small epsilon to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # UCB parameter: balance between exploration and exploitation
    # For minimization, we want to find low values
    # UCB for minimization: -mu + beta * sigma
    # We use an adaptive beta based on the number of observations
    # Since we don't have explicit n, we use a fixed reasonable value
    
    # Use a cost-aware UCB: we penalize high costs
    # Score = (potential improvement) / cost
    # Potential improvement for minimization: how much lower this point could be
    
    # Approach: Use expected hypervolume improvement or a simple cost-aware UCB
    
    # Simple approach: UCB score normalized by cost
    # For minimization, lower mu is better, higher sigma means more uncertainty
    # We want to explore points with high sigma (uncertainty) and low mu (potential)
    
    # UCB for minimization: -mu + beta * sigma
    # This gives higher scores to points that are either low mean or high uncertainty
    
    beta = 1.0  # Exploration-exploitation trade-off
    
    # UCB score for minimization
    ucb_score = -mu + beta * sigma_safe
    
    # Make it cost-aware: divide by cost
    # But we need to ensure scores are finite and well-scaled
    scores = ucb_score / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
