"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition function.
    
    Instead of computing a deterministic score like EI, we sample from the
    posterior predictive distribution at each candidate point. The candidate
    with the lowest sampled value is selected (since we are minimizing).
    
    To return a single score array where "higher is better", we return the
    negative of the sampled values. This effectively ranks candidates by
    their potential to improve the minimum.
    
    Using a single sample per call is standard for Thompson Sampling in this
    context. The randomness is inherent to the method.
    """
    # Ensure sigma is positive to avoid division/log issues if used, though
    # for normal sampling we just need mu and sigma.
    # sigma is already standard deviation.
    
    # Generate a random sample from the normal distribution N(mu, sigma)
    # We use np.random.normal with loc=mu and scale=sigma.
    # This is vectorized and fast.
    
    # Note: The evaluator calls this function once per step.
    # The "score" returned is used to pick the max.
    # If we return -sample, the max of -sample is the min of sample.
    
    samples = np.random.normal(loc=mu, scale=sigma)
    
    # Return negative samples so that the highest score corresponds to the lowest value
    score = -samples
    
    # Ensure finiteness (normal distribution samples are almost surely finite,
    # but guard against extreme edge cases or NaNs in input)
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
# EVOLVE-BLOCK-END
