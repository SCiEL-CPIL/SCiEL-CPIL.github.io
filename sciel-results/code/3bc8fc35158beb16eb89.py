"""Acquisition function for Bayesian optimization of ackley 2d."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_DEADLINE_MARGIN = 10.0  # seconds before the 330s wall-clock kill
_TOTAL_BUDGET = 330.0


def _time_remaining():
    return _TOTAL_BUDGET - _DEADLINE_MARGIN - (time.time() - _start_time)


def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition function for minimizing Ackley 2D.
    
    Draws a sample from the posterior at each candidate and returns
    negative sample values as scores (higher = more likely to be minimum).
    
    Parameters:
        mu: array of shape (2000,) - posterior mean
        sigma: array of shape (2000,) - posterior std
        f_best: float - best observed value so far
    
    Returns:
        scores: array of shape (2000,) - higher means evaluate next
    """
    # Deadline guard: if time is running out, return pure exploitation
    if _time_remaining() < 5.0:
        return -mu.copy()
    
    # Thompson Sampling: draw one sample from the posterior
    # The sample is mu + sigma * z where z ~ N(0,1)
    # We want to find the minimum, so we score as -sample
    # (higher score = lower sampled value = more promising)
    
    # Use a fixed random seed based on the call pattern to ensure
    # reproducibility within a run but different across runs
    # Actually, we want randomness for exploration, so use default RNG
    z = np.random.randn(len(mu))
    
    # Sample from posterior
    sample = mu + sigma * z
    
    # Score: negative of sample (we want to minimize, so lower sample = better)
    scores = -sample
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
