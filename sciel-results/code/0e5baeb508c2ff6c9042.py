"""Rewritten acquisition for Bayesian optimization of a 2D RBF mixture."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Compute a robust acquisition score for each candidate.
    
    We use a combination of expected improvement and a small exploration term
    to avoid numerical issues and encourage exploration in high-uncertainty
    regions. The score is higher for points we want to evaluate next.
    """
    # Clip sigma to avoid division by zero or extremely small values
    sigma = np.maximum(sigma, 1e-8)
    
    # Expected Improvement (EI)
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Add a small exploration bonus based on uncertainty
    # This helps when EI is flat (e.g., early in optimization)
    exploration_bonus = 0.1 * sigma
    
    # Combine EI and exploration bonus
    score = ei + exploration_bonus
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
