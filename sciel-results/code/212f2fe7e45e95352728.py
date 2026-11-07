"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining UCB exploration with expected improvement.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean predictions
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best observed value so far
    cost : ndarray, shape (2000,)
        Evaluation costs for each candidate
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = more promising)
    """
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle edge cases
    n = len(mu)
    scores = np.zeros(n, dtype=np.float64)
    
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)
    
    # Expected Improvement (EI)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # UCB term with adaptive beta based on uncertainty
    # Use a moderate exploration constant
    beta = 0.5
    
    # Combine EI and UCB, penalize by cost
    # Weight EI more heavily as we get closer to the optimum
    # Use a power penalty on cost to balance exploration vs. cost
    ucb = -mu + beta * sigma_safe
    
    # Normalize costs to [0, 1] range for better scaling
    cost_min = np.min(cost_safe)
    cost_max = np.max(cost_safe)
    if cost_max > cost_min:
        cost_norm = (cost_safe - cost_min) / (cost_max - cost_min)
    else:
        cost_norm = np.ones_like(cost_safe)
    
    # Blend EI and UCB, with cost penalty
    # Higher cost should reduce the score
    score = (ei + 0.3 * ucb) / (cost_norm + 1.0)
    
    # Ensure finite output
    scores = np.where(np.isfinite(score), score, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
