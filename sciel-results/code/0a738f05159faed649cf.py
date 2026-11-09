"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Beale 2D.
    Uses a combination of Expected Improvement and Uncertainty (UCB-like)
    to balance exploitation and exploration.
    """
    # Ensure sigma is positive to avoid division by zero
    sigma = np.maximum(sigma, 1e-9)
    
    # Expected Improvement (EI)
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Uncertainty term (UCB)
    # We add a small uncertainty bonus to encourage exploration
    # The coefficient is tuned to be small so EI dominates when improvement is clear
    ucb = 0.5 * sigma
    
    # Combine EI and UCB
    # We normalize EI to be comparable with UCB
    # Actually, just summing them is a common heuristic
    # Let's try a weighted sum
    score = ei + ucb
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
