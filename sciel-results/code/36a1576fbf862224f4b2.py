"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Expected Improvement (EI) with a small exploration bonus via
    # adding a small fraction of sigma to f_best in the numerator,
    # which effectively increases exploration in low-certainty regions.
    #
    # Standard EI: (f_best - mu) * Phi(z) + sigma * phi(z), where z = (f_best - mu)/sigma
    #
    # We use a slightly modified version that adds a small exploration term
    # to encourage exploring regions with high uncertainty.
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # Standard EI
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Add a small exploration bonus proportional to sigma
    # This helps escape local optima by giving a boost to high-uncertainty regions
    exploration_weight = 0.1
    exploration_bonus = exploration_weight * sigma_safe
    
    scores = ei + exploration_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
