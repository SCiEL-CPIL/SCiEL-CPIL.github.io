"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Pareto-frontier guided exploration strategy.
    
    1. Identify the candidate with the lowest predicted mean (mu_min) as an aggressive exploitation anchor.
    2. Compute the acquisition score as the product of sigma and the normalized distance from this anchor in prediction space.
    
    This geometric approach prioritizes high-uncertainty regions that are also far from the current best prediction,
    balancing exploration and exploitation.
    """
    # Handle edge cases where sigma might be zero or very small
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Step 1: Find the anchor - candidate with lowest predicted mean
    mu_min = np.min(mu)
    
    # Step 2: Compute normalized distance from the anchor in prediction space
    # The distance is |mu - mu_min|, normalized by the range of mu to make it scale-invariant
    mu_range = np.max(mu) - mu_min
    if mu_range < 1e-9:
        # All predictions are essentially the same
        normalized_distance = np.ones_like(mu)
    else:
        normalized_distance = (mu - mu_min) / mu_range
    
    # Step 3: Compute acquisition score as product of sigma and normalized distance
    # This rewards points that have both high uncertainty AND are far from the exploitation anchor
    score = sigma_safe * normalized_distance
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
