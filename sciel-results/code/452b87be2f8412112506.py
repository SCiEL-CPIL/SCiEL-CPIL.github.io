"""Acquisition function for Bayesian optimization of michalewicz 2d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining Expected Improvement with an adaptive exploration term.
    
    The adaptive xi parameter scales with the posterior uncertainty to balance
    exploration and exploitation. In high-uncertainty regions, xi increases,
    encouraging exploration. Near the current best, xi decreases, focusing on exploitation.
    """
    # Compute the expected improvement (EI) with adaptive xi
    # xi scales with the mean sigma to encourage exploration in uncertain regions
    mean_sigma = np.mean(sigma)
    # Clamp xi to a reasonable range to avoid extreme values
    xi = 0.01 + 0.1 * (mean_sigma / (np.std(sigma) + 1e-9))
    xi = np.clip(xi, 0.0, 0.5)
    
    # Standardized improvement
    z = (f_best - mu - xi) / (sigma + 1e-9)
    
    # Expected improvement
    ei = (f_best - mu - xi) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Add a small exploration bonus proportional to sigma to help escape local minima
    # This term is small enough not to dominate but large enough to encourage diversity
    exploration_bonus = 0.05 * sigma
    
    return ei + exploration_bonus
# EVOLVE-BLOCK-END
