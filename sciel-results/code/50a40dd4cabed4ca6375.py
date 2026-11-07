"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_start_time = time.time()
_deadline = _start_time + 285.0

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of ackley 2d.
    Uses Thompson Sampling: draw one sample from the predictive posterior
    for each candidate and score by the negative sample. This naturally
    balances exploration and exploitation.
    """
    # Deadline guard
    if time.time() > _deadline:
        return -mu

    n = len(mu)
    
    # Ensure sigma is non-negative and finite
    sigma = np.maximum(sigma, 0.0)
    sigma = np.nan_to_num(sigma, nan=0.0, posinf=0.0, neginf=0.0)
    mu = np.nan_to_num(mu, nan=f_best, posinf=f_best, neginf=f_best)
    
    # Thompson Sampling: sample from N(mu_i, sigma_i^2) for each candidate
    # The score is -sample, so lower predicted values get higher scores
    # This naturally explores where sigma is large and exploits where mu is low
    
    # To make this deterministic across calls (important for reproducibility in the evaluator),
    # we use a fixed random seed derived from the function inputs
    # Actually, for Thompson Sampling to work properly in a loop, we need fresh randomness
    # each time it's called. But the evaluator calls it once per step.
    # Let's use np.random with a seed based on the current time or a global counter
    # to ensure different samples each call.
    
    # Use a global counter to ensure different random seeds each call
    global _ts_counter
    if not hasattr(acquisition, '_ts_counter'):
        acquisition._ts_counter = 0
    acquisition._ts_counter += 1
    
    # Create a random state with a seed that changes each call
    rng = np.random.RandomState(acquisition._ts_counter * 7919 + 12345)
    
    # Draw samples from the predictive posterior
    # For points with very small sigma, the sample is close to mu
    # For points with large sigma, the sample has more variance
    samples = mu + sigma * rng.standard_normal(n)
    
    # Score: higher means evaluate next, so we want low predicted values
    scores = -samples
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
