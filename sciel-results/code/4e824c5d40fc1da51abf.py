"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    UCB-based acquisition function for Bayesian optimization.
    
    Args:
        mu: array of shape (2000,), posterior mean
        sigma: array of shape (2000,), posterior standard deviation
        f_best: float, best observed value so far
    
    Returns:
        array of shape (2000,), acquisition scores (higher = better)
    """
    # Ensure numerical stability
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # UCB: balance exploitation (low mu) and exploration (high sigma)
    # For minimization, we want low mu and high uncertainty
    # UCB = -mu + beta * sigma (negative because we minimize)
    
    # Adaptive beta: start with higher exploration, decrease as we get more evaluations
    # Since we don't know the exact iteration, use a reasonable fixed beta
    # For Weierstrass 1D with many local minima, moderate exploration is key
    beta = 1.5
    
    # UCB score for minimization: we want to maximize (-mu + beta * sigma)
    # This encourages both low predicted values and high uncertainty
    score = -mu + beta * sigma_safe
    
    # Add a small exploitation component based on EI to ensure we don't ignore
    # very promising points with low uncertainty
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Combine: primarily UCB, with a small EI component
    # Weight: 70% UCB, 30% EI
    ucb_normalized = score - np.min(score)
    ucb_range = np.max(ucb_normalized) - np.min(ucb_normalized) + 1e-9
    ucb_norm = ucb_normalized / ucb_range
    
    ei_normalized = ei - np.min(ei)
    ei_range = np.max(ei_normalized) - np.min(ei_normalized) + 1e-9
    ei_norm = ei_normalized / ei_range
    
    combined = 0.7 * ucb_norm + 0.3 * ei_norm
    
    return combined
# EVOLVE-BLOCK-END
