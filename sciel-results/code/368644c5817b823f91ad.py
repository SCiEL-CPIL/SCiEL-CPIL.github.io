"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware UCB acquisition function for minimization.
    
    Args:
        mu: Array of shape (N,) with predicted means.
        sigma: Array of shape (N,) with predicted standard deviations.
        f_best: Scalar, the best observed value so far.
        cost: Array of shape (N,) with evaluation costs.
        
    Returns:
        Array of shape (N,) with acquisition scores (higher is better).
    """
    # UCB for minimization: We want to maximize the potential improvement.
    # The lower confidence bound is mu - beta * sigma.
    # We want this bound to be as low as possible.
    # So we maximize: f_best - (mu - beta * sigma)
    
    beta = 1.0
    
    # Calculate the acquisition value
    # f_best - mu + beta * sigma
    scores = f_best - mu + beta * sigma
    
    # Make cost-aware by dividing by cost
    # Cost is always >= 1, so no division by zero risk, but add epsilon for safety
    epsilon = 1e-8
    cost_safe = cost + epsilon
    
    scores = scores / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
