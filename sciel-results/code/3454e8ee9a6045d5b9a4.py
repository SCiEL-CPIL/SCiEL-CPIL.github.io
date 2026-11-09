"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization.
    Uses Expected Improvement (EI) with a small exploration parameter to balance
    exploitation and exploration.
    """
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute z-score for EI
    z = (f_best - mu) / sigma_safe
    
    # Standard Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Add a small amount of uncertainty weighting to encourage exploration
    # This helps escape local optima and explore more of the search space
    # The term sigma_safe * norm.pdf(z) is already part of EI, but we can
    # add a small bonus for high uncertainty regions
    exploration_bonus = 0.01 * sigma_safe * norm.pdf(z)
    
    return ei + exploration_bonus
# EVOLVE-BLOCK-END
