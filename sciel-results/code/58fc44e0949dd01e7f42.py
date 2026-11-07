"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global timer for deadline guard
_START_TIME = time.time()
# The evaluator kills after 330s. We use 320s as a safety margin.
_DEADLINE = 320.0

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Probability of Improvement (PI).
    
    Args:
        mu: Posterior mean at candidate points (shape 2000,).
        sigma: Posterior standard deviation at candidate points (shape 2000,).
        f_best: Lowest observed value so far.
        
    Returns:
        Scores for the candidate points (shape 2000,). Higher is better.
    """
    # Deadline guard
    # If we are close to the time limit, return a simple heuristic to ensure 
    # the function returns quickly and doesn't contribute to a timeout.
    # -mu is a greedy choice that is computationally trivial.
    if time.time() - _START_TIME > _DEADLINE:
        return -mu

    # Calculate z-score for Probability of Improvement
    # z = (f_best - mu) / sigma
    # We add a small epsilon to sigma to prevent division by zero
    # if the GP is extremely confident (sigma ~ 0).
    epsilon = 1e-9
    safe_sigma = sigma + epsilon
    
    z = (f_best - mu) / safe_sigma
    
    # The acquisition score is the probability that f(x) < f_best
    # P(f(x) < f_best) = Phi(z)
    scores = norm.cdf(z)
    
    return scores
# EVOLVE-BLOCK-END
