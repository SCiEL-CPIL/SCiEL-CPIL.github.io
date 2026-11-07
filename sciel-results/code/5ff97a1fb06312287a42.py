"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    # Structural change: Use a lower quantile of the posterior to target the global lower bound.
    # Score is designed to be maximized.
    # We approximate the expected minimum of a large number of samples from the posterior
    # by the lower quantile Q_alpha = mu + z_alpha * sigma.
    # To maximize the chance of finding a low value, we maximize -Q_alpha = -mu - z_alpha * sigma.
    # Since z_alpha is negative for small alpha, -z_alpha is positive.
    # Let k = -z_alpha. For alpha = 1e-4, z_alpha ~ -3.7.
    # We use k = 3.0 to balance exploration and exploitation, slightly less aggressive than the 1e-4 quantile.
    k = 3.0
    # Ensure sigma is non-negative for safety, though it should be from GP
    sigma = np.maximum(sigma, 1e-9)
    
    # Higher score means better candidate
    scores = -mu + k * sigma
    
    # Return finite scores
    return scores
# EVOLVE-BLOCK-END
