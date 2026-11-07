"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    """
    Marginal Thompson Sampling acquisition function.
    
    Samples from the posterior distribution at each candidate point independently.
    Returns the negative of the sampled value, so that higher scores correspond
    to lower expected function values (minimization).
    """
    # Ensure sigma is positive to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Generate standard normal random numbers
    z = np.random.randn(len(mu))
    
    # Sample from the posterior: f(x) ~ N(mu, sigma^2)
    samples = mu + sigma_safe * z
    
    # We want to find the minimum.
    # Higher acquisition score -> more likely to be chosen.
    # So we invert the sign.
    score = -samples
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
