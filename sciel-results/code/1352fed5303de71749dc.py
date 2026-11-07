"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_start_time = time.time()
_deadline = _start_time + 290.0

def acquisition(mu, sigma, f_best):
    # Deadline guard: if we are running out of time, fall back to greedy (minimize mu)
    if time.time() > _deadline:
        return -mu
    
    # Ensure sigma is strictly non-negative and finite
    sigma = np.maximum(sigma, 0.0)
    sigma = np.nan_to_num(sigma, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Ensure mu is finite
    mu = np.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Thompson Sampling: sample from the predictive posterior and pick the candidate
    # with the lowest sampled value. We return negative sampled values as scores
    # (higher score = lower predicted objective value).
    
    # Generate a single sample from N(mu, sigma^2) for each candidate
    sigma_safe = np.where(sigma > 1e-12, sigma, 1e-12)
    samples = mu + sigma_safe * np.random.standard_normal(len(mu))
    
    # Score is negative of the sample (higher score means lower predicted value)
    scores = -samples
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
