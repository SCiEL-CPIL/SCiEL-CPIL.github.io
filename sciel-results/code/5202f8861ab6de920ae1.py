import numpy as np

def acquisition(mu, sigma, f_best):
    # Thompson Sampling for minimization
    # Draw a single sample from the Gaussian posterior N(mu, sigma^2) for each candidate.
    # For minimization, we want to find the lowest value, so we minimize the sampled value.
    # The task requires returning scores where HIGHER means evaluate next.
    # Therefore, we return the negative of the sampled value.
    
    # Generate a single standard normal random vector of size 2000
    # This single vector is reused for all candidates, which is the standard 
    # efficient implementation of Thompson Sampling (common random numbers).
    z = np.random.standard_normal(len(mu))
    
    # Sample from the posterior
    # sample = mu + sigma * z
    # Score = -sample = -mu - sigma * z
    
    # Ensure numerical stability: sigma should be non-negative, but just in case
    # Also, f_best is not strictly needed for the score calculation in pure TS,
    # but it's part of the signature.
    
    return -mu - sigma * z
