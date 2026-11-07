"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    # Use Expected Improvement (EI) with a small exploration bonus based on
    # uncertainty, which can help escape local optima in 1D problems.
    # EI is a strong baseline for Bayesian optimization.
    
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standardized improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Add a small exploration term: prefer points with higher uncertainty
    # when they are not clearly worse than the best
    # This is a soft version of UCB combined with EI
    # We use a small weight to keep EI dominant but add exploration
    exploration_weight = 0.05
    scores = ei + exploration_weight * sigma
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
