"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition function.
    Samples from the posterior distribution and selects the point with the lowest sampled value.
    Returns scores where higher is better for selection.
    """
    # Ensure sigma is strictly positive to avoid division by zero or undefined behavior
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Draw samples from the standard normal distribution
    # This introduces stochasticity, balancing exploration and exploitation
    z = np.random.normal(0, 1, size=mu.shape)
    
    # Compute the sampled function values
    f_sampled = mu + z * sigma_safe
    
    # We want to minimize the function value.
    # The acquisition function must return scores where HIGHER is better.
    # Therefore, we return the negative of the sampled value.
    scores = -f_sampled
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
