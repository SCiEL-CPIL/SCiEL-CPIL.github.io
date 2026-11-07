"""Acquisition function for Bayesian optimization of hartmann 3d."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    """
    Upper Confidence Bound (UCB) acquisition function for minimization.
    
    Parameters:
    - mu: array of shape (N,) with posterior mean
    - sigma: array of shape (N,) with posterior standard deviation
    - f_best: scalar, the lowest observed value so far
    
    Returns:
    - scores: array of shape (N,) with acquisition scores (higher is better)
    """
    # Clip sigma to avoid division by zero and ensure positivity
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # UCB for minimization:
    # We want to minimize f(x).
    # The predicted value is mu.
    # The uncertainty is sigma.
    # We want to find points where the lower bound of the confidence interval is low.
    # Lower bound = mu - beta * sigma.
    # To maximize the "desirability", we can use - (mu - beta * sigma) = -mu + beta * sigma.
    # Alternatively, some formulations use mu + beta * sigma for maximization.
    # For minimization, we want low mu.
    # So, score = -mu + beta * sigma.
    
    beta = 2.0
    scores = -mu + beta * sigma_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
