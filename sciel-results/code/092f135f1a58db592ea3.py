"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition function.
    
    Args:
        mu: Array of means (2000,)
        sigma: Array of standard deviations (2000,)
        f_best: Current best observed value (scalar)
        
    Returns:
        Array of scores (2000,). Higher score indicates the candidate 
        is more likely to be the next evaluation point (i.e., has a lower 
        sampled function value).
    """
    # Ensure sigma is positive to avoid numerical issues in sampling
    sigma = np.maximum(sigma, 1e-9)
    
    # Draw a sample from the posterior distribution N(mu, sigma^2) for each candidate
    # This balances exploration (high sigma -> high variance in samples) 
    # and exploitation (low sigma -> samples close to mu).
    samples = np.random.normal(loc=mu, scale=sigma)
    
    # The goal is to minimize the function. 
    # The evaluator selects the candidate with the highest score.
    # Therefore, we return the negative of the samples so that 
    # the lowest sampled value (best potential minimizer) gets the highest score.
    return -samples
# EVOLVE-BLOCK-END
