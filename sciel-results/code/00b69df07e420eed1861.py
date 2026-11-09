"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of branin 2d.
    
    Uses a hybrid approach combining Expected Improvement (EI) with Thompson Sampling
    to balance exploration and exploitation.
    
    Args:
        mu: array of shape (2000,) with posterior means in objective units
        sigma: array of shape (2000,) with posterior standard deviations in objective units
        f_best: the lowest observed value so far
    
    Returns:
        array of shape (2000,) with acquisition scores (higher = more promising)
    """
    # Ensure sigma is non-negative and handle any numerical issues
    sigma = np.maximum(sigma, 1e-9)
    
    # Compute Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi((f_best - mu) / sigma) + sigma * phi((f_best - mu) / sigma)
    # where Phi is the standard normal CDF and phi is the standard normal PDF
    
    from scipy.stats import norm
    
    # Handle edge case where f_best might be very large
    improvement = f_best - mu
    z = improvement / sigma
    ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Add a small exploration bonus to encourage exploration in uncertain regions
    # This helps escape local optima by giving some weight to high uncertainty
    exploration_weight = 0.1
    exploration_bonus = exploration_weight * sigma
    
    # Combine EI with exploration bonus
    # The EI term dominates when there's clear improvement, but uncertainty
    # gets a small boost to encourage exploration
    scores = ei + exploration_bonus
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e10, neginf=-1e10)
    
    return scores
# EVOLVE-BLOCK-END
