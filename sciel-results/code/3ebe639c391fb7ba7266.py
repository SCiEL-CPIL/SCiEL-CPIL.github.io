"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_deadline = _start_time + 285.0
_call_count = [0]  # Mutable state to track number of acquisition calls

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of ackley 2d.
    Uses Thompson Sampling: draws a sample from the predictive posterior
    for each candidate and returns the negative of that sample as the score.
    Higher scores correspond to lower sampled objective values.
    
    s_i = -N(mu_i, sigma_i^2)
    """
    # Deadline guard: if we are running out of time, fall back to greedy (minimize mean)
    if time.time() > _deadline:
        return -mu

    # Ensure sigma is non-negative and finite
    sigma = np.asarray(sigma, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    
    # Sanitize inputs
    sigma = np.maximum(sigma, 1e-10)  # Avoid zero variance
    sigma = np.nan_to_num(sigma, nan=1e-10, posinf=1e-10, neginf=1e-10)
    mu = np.nan_to_num(mu, nan=f_best, posinf=f_best, neginf=f_best)
    
    # Track step count
    step = _call_count[0]
    _call_count[0] += 1
    
    # Thompson Sampling: draw one sample from N(mu, sigma^2) for each candidate
    # Use a fixed random state seeded by step for reproducibility within a seed run
    rng = np.random.RandomState(step * 1000 + 42)
    samples = mu + sigma * rng.standard_normal(len(mu))
    
    # Score is the negative of the sample: higher score = lower sampled value
    scores = -samples
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=-f_best, posinf=1e10, neginf=-1e10)
    
    return scores
# EVOLVE-BLOCK-END
