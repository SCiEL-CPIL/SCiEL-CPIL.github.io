"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    # Thompson Sampling:
    # Sample from the Gaussian posterior at each candidate point.
    # The point with the lowest sampled value is the most likely to be the new minimum.
    # We return negative samples so that the highest score corresponds to the lowest sampled value.
    
    # Ensure sigma is positive to avoid issues with normal distribution std
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Draw one sample from N(mu, sigma) for each candidate
    # np.random.normal takes size as the last argument
    samples = np.random.normal(mu, sigma_safe)
    
    # We want to maximize the score, so we negate the sampled values
    # The evaluator picks the argmax of the returned scores.
    return -samples
# EVOLVE-BLOCK-END
