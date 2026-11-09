"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition function for Bayesian optimization of Hartmann 6d.
    Samples one value from the posterior Gaussian for each candidate and returns
    the sampled values (negated so higher score = lower sampled objective).
    This naturally balances exploration and exploitation.
    """
    # Deadline guard
    t_start = time.time()
    
    # Ensure sigma is strictly positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Thompson Sampling: draw one sample from N(mu, sigma) for each candidate
    # Use a fixed random seed based on the hash of mu to ensure reproducibility
    # within a single call but vary across calls
    # Actually, we want fresh randomness each call. Use np.random default.
    # But we need to be careful: the evaluator calls this 25 times per seed (30-5=25 steps).
    # We just draw fresh samples each time.
    
    # Draw samples from the posterior
    samples = mu + sigma_safe * np.random.randn(mu.shape[0])
    
    # We want to minimize the objective, so we want to pick the candidate with the
    # lowest sampled value. Since the acquisition function should return scores where
    # HIGHER means "evaluate next", we negate the samples.
    scores = -samples
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Safety check for deadline
    if time.time() - t_start > 250:
        return scores
        
    return scores
# EVOLVE-BLOCK-END
