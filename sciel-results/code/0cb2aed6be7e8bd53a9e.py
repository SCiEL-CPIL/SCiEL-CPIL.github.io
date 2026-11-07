"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Expected improvement with a distance-to-best repulsive weighting.
    # We approximate distance using the posterior mean and std to create
    # a repulsive force from regions where the GP is confident (low sigma)
    # and attractive toward uncertain regions (high sigma).
    
    # Basic EI
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Distance-based repulsion: scale EI by uncertainty relative to the 
    # maximum uncertainty, creating a preference for unexplored regions.
    # This helps escape local minima in multimodal functions like Michalewicz.
    
    # Normalize uncertainty to [0, 1] range
    sigma_max = np.max(sigma_safe)
    sigma_min = np.min(sigma_safe)
    if sigma_max > sigma_min + 1e-12:
        uncertainty_weight = (sigma_safe - sigma_min) / (sigma_max - sigma_min)
    else:
        uncertainty_weight = np.ones_like(sigma_safe)
    
    # Combine EI with uncertainty-driven exploration
    # Higher uncertainty regions get boosted scores to encourage exploration
    combined = ei * (1.0 + 0.5 * uncertainty_weight)
    
    # Ensure finite values
    combined = np.nan_to_num(combined, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return combined
# EVOLVE-BLOCK-END
