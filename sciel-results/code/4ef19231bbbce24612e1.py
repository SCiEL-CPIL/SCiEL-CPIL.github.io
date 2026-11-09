"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    # Thompson Sampling: draw a single sample from the posterior for each candidate
    # and return negative sampled values (lower objective is better, so higher score = more promising).
    # sigma is the standard deviation.
    # We sample: x ~ N(mu, sigma^2)
    # Return -x as the acquisition score (higher means we want to evaluate that candidate).
    
    # Use standard normal random numbers scaled by sigma and shifted by mu.
    # Single draw per candidate (vectorized over 2000 candidates).
    eps = np.random.standard_normal(mu.shape)
    samples = mu + sigma * eps
    
    # We want to minimize, so lower sampled value is better.
    # Return negative of the sample so that higher score = lower sampled objective.
    return -samples
# EVOLVE-BLOCK-END
