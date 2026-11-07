"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    -----------
    mu : array of shape (2000,) - GP mean predictions
    sigma : array of shape (2000,) - GP standard deviation predictions
    f_best : float - best observed value so far
    cost : array of shape (2000,) - cost of evaluating each candidate
    
    Returns:
    --------
    scores : array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle edge cases
    n = len(mu)
    if n == 0:
        return np.array([])
    
    # Clip sigma to avoid division by zero
    sigma = np.maximum(sigma, 1e-8)
    
    # Clip cost to avoid division by zero
    cost = np.maximum(cost, 1e-8)
    
    # Compute z-score
    z = (f_best - mu) / sigma
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # UCB (Upper Confidence Bound) - for minimization, we want lower mu + exploration
    # Beta controls exploration; for Hartmann 6D, moderate exploration helps
    beta = 2.0
    ucb = mu - beta * sigma  # For minimization, lower is better, so we subtract
    
    # Blend EI and UCB
    # Standardize both to have zero mean and unit variance for robust blending
    ei_mean = np.mean(ei)
    ei_std = np.std(ei) + 1e-8
    ei_std = np.where(ei_std < 1e-8, 1.0, ei_std)
    
    ucb_mean = np.mean(ucb)
    ucb_std = np.std(ucb) + 1e-8
    ucb_std = np.where(ucb_std < 1e-8, 1.0, ucb_std)
    
    ei_norm = (ei - ei_mean) / ei_std
    ucb_norm = (ucb - ucb_mean) / ucb_std
    
    # Blend with equal weight
    blended = 0.5 * ei_norm + 0.5 * ucb_norm
    
    # Convert back to original scale for cost normalization
    # The blended score is in standardized units; we need to normalize by cost
    # Use the mean of the original EI as a scale factor
    scale = np.abs(ei_mean) + 1e-8
    
    # Cost-aware: divide by cost
    scores = blended * scale / cost
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
