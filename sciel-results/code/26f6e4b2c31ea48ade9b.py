"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition: sample a single value from the posterior
    at each candidate point and return the negative of that sample.
    Higher score = lower predicted value = more likely to be evaluated next.
    """
    # Handle any non-positive sigma values to avoid sampling issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Draw a single sample from N(mu, sigma) at each point
    # We use a deterministic-ish approach: since we can't pass a seed,
    # we just draw from the standard normal and scale
    # This is a single sample Thompson sampling step
    z = np.random.standard_normal(len(mu))
    sample = mu + sigma_safe * z
    
    # We want to minimize, so higher score means lower predicted value
    # Return negative of the sample
    return -sample
# EVOLVE-BLOCK-END
