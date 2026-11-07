"""Acquisition for Bayesian optimization of Michalewicz 2D using Thompson Sampling."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = time.time() + 290.0  # safety margin before 300s kill

def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition function for Bayesian optimization (minimization).
    Samples from the posterior N(mu, sigma^2) at each candidate and returns
    negative samples as scores. Higher score = lower sampled value = evaluate next.
    
    This naturally balances exploration and exploitation:
    - High sigma regions have more variable samples, encouraging exploration
    - Low sigma regions have samples clustered near mu, encouraging exploitation
    """
    # Deadline guard: fall back to pure exploitation if running out of time
    if time.time() > _DEADLINE:
        return -mu
    
    # Sanitize inputs
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Thompson Sampling: draw one sample from N(mu, sigma) for each candidate
    # The evaluator selects the candidate with the highest score (lowest sampled value)
    rng = np.random.default_rng()
    samples = rng.normal(loc=mu, scale=sigma_safe)
    
    # Return negative samples: higher score means lower (better) sampled value
    scores = -samples
    
    # Ensure finite scores
    scores[~np.isfinite(scores)] = -1e10
    
    return scores
# EVOLVE-BLOCK-END
