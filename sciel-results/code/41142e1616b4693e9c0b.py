"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Upper Confidence Bound (UCB) for minimization.
    
    UCB = mu - kappa * sigma
    
    This balances exploitation (low mu) with exploration (high sigma).
    We use a fixed kappa that works well for this problem.
    """
    # For minimization, we want to minimize mu - kappa * sigma
    # So the acquisition score should be negative of that (higher is better)
    # Or equivalently, we return -mu + kappa * sigma
    
    # Let's use a kappa that balances exploration and exploitation.
    # A common choice is kappa = 1.0 or based on log(n) but since we don't have n,
    # we use a fixed value.
    kappa = 1.0
    
    # Handle small sigma to avoid numerical issues
    eps = 1e-12
    safe_sigma = np.maximum(sigma, eps)
    
    # UCB for minimization: we want to minimize mu - kappa * sigma
    # So the score (higher is better) is: -mu + kappa * sigma
    ucb_score = -mu + kappa * safe_sigma
    
    # Ensure all values are finite
    ucb_score = np.where(np.isfinite(ucb_score), ucb_score, 0.0)
    
    return ucb_score
# EVOLVE-BLOCK-END
