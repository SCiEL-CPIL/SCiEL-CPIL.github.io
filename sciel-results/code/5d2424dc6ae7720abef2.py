"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_deadline = _start_time + 285.0

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of ackley 2d.
    Uses Thompson Sampling: draws a sample from the predictive posterior
    for each candidate and returns the negative of that sample as the score.
    Higher scores correspond to lower sampled objective values.
    """
    # Deadline guard: if we are running out of time, fall back to greedy (minimize mean)
    if time.time() > _deadline:
        return -mu

    n = len(mu)
    
    # Ensure sigma is non-negative and finite
    # The task states sigma is a standard deviation.
    sigma = np.asarray(sigma)
    mu = np.asarray(mu)
    
    # Sanitize inputs
    sigma = np.maximum(sigma, 1e-10) # Avoid zero variance
    sigma = np.nan_to_num(sigma, nan=1e-10, posinf=1e-10, neginf=1e-10)
    mu = np.nan_to_num(mu, nan=f_best, posinf=f_best, neginf=f_best)
    
    # Thompson Sampling:
    # For each candidate i, sample y_i ~ N(mu_i, sigma_i^2)
    # We want to minimize the objective, so we want to pick the candidate
    # with the smallest sampled value.
    # The acquisition function must return scores where HIGHER is better.
    # Therefore, we return -y_i.
    
    # Use a fixed random state? No, the evaluator calls this multiple times.
    # Using the global random state is standard for TS in this context.
    # However, to be safe and reproducible across different runs of the same seed,
    # one might worry about RNG state. But the task says "for each seed 0, 1 and 2",
    # implying the evaluator handles the main seed. The acquisition function
    # itself should just use the current RNG state.
    
    samples = np.random.normal(loc=mu, scale=sigma, size=n)
    
    # Ensure finite scores
    scores = -samples
    scores = np.nan_to_num(scores, nan=-f_best, posinf=1e10, neginf=-1e10)
    
    return scores
# EVOLVE-BLOCK-END
