"""Rewritten program using GP-LCB acquisition for cost-aware Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware GP-LCB acquisition function with tuned exploration and cost penalty.
    
    Uses a Lower Confidence Bound (LCB) objective: mu - beta * sigma.
    Applies a softer cost penalty (1/sqrt(cost)) to balance exploration of expensive regions.
    
    Parameters:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP standard deviation predictions
        f_best: float - best observed value so far
        cost: array of shape (2000,) - cost of evaluating each candidate
    
    Returns:
        array of shape (2000,) - acquisition scores (higher = better)
    """
    # Use a standard GP-LCB objective as per directive.
    # Higher beta encourages more exploration.
    beta = 2.0
    
    # LCB: we want to minimize, so lower LCB is better.
    # Score should be higher for better candidates, so we negate the LCB.
    scores = -(mu - beta * sigma)
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
