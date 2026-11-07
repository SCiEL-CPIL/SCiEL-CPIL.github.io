"""Acquisition function for Bayesian optimization of hartmann 3d."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    """
    Lower Confidence Bound (LCB) acquisition function for minimization.
    Score = -LCB = -mu + beta * sigma.
    Higher score means more promising (lower expected value or high uncertainty).
    
    Parameters:
    - mu: array of shape (N,) with posterior mean
    - sigma: array of shape (N,) with posterior standard deviation
    - f_best: scalar, the lowest observed value so far (unused in standard LCB but kept for interface)
    
    Returns:
    - scores: array of shape (N,) with acquisition scores (higher is better)
    """
    # Clip sigma to avoid numerical issues, though LCB doesn't divide by it
    sigma_safe = np.maximum(sigma, 0.0)
    
    # Beta parameter controls the trade-off between exploration and exploitation.
    # A higher beta encourages more exploration.
    # For a small budget (30 evals), a moderate beta is good.
    beta = 2.0
    
    # LCB for minimization: mu - beta * sigma
    # We want to maximize the score, so we return -LCB = -mu + beta * sigma
    scores = -mu + beta * sigma_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
