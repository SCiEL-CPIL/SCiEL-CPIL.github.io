"""Cost-aware acquisition function for Bayesian optimization (hartmann 6d)."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-normalized Lower Confidence Bound (LCB) acquisition function.
    
    Args:
        mu: array of shape (2000,) - posterior mean
        sigma: array of shape (2000,) - posterior standard deviation
        f_best: scalar - best observed value
        cost: array of shape (2000,) - evaluation cost for each candidate
    
    Returns:
        array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    # Ensure sigma is non-negative and handle numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # LCB: mu - beta * sigma
    # Choose beta to balance exploration and exploitation
    # 2.0 is a standard robust choice for multi-dimensional optimization
    beta = 2.0
    
    # Compute LCB values (lower is better for the objective)
    lcb = mu - beta * sigma_safe
    
    # We want to maximize the acquisition score, so we want to minimize LCB / cost
    # Since we return higher scores for better candidates, we negate LCB and divide by cost
    # Score = -LCB / cost = (beta * sigma - mu) / cost
    # This is equivalent to maximizing (beta * sigma - mu) / cost
    
    # Ensure cost is positive to avoid division by zero
    cost_safe = np.maximum(cost, 1e-9)
    
    # Compute the cost-normalized negative LCB
    scores = (beta * sigma_safe - mu) / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
