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
    
    Instead of using a deterministic acquisition function like LCB or EI,
    we draw samples from the posterior distribution and select the candidate
    with the lowest sampled value. This naturally balances exploration and
    exploitation and is robust to the specific tuning of exploration parameters.
    
    Parameters:
        mu: array of shape (2000,) - posterior mean
        sigma: array of shape (2000,) - posterior std
        f_best: float - best observed value so far
    
    Returns:
        scores: array of shape (2000,) - higher means evaluate next
    """
    # Check time budget
    if _time_remaining() < 5.0:
        # Fallback to pure exploitation if time is critical
        return -mu.copy()
    
    # Number of posterior samples to draw
    # 10 samples is a good trade-off between computational cost and diversity
    n_samples = 10
    
    # Generate random normal variables for the samples
    # Shape: (n_samples, 2000)
    z = np.random.normal(0, 1, size=(n_samples, len(mu)))
    
    # Compute samples: y = mu + sigma * z
    # mu and sigma are broadcasted to (n_samples, 2000)
    samples = mu[None, :] + sigma[None, :] * z
    
    # For minimization, we want to find the candidate that has the lowest value
    # in at least one sample (or generally low expected minimum).
    # Taking the minimum over samples gives a score that favors candidates
    # that are low in some plausible scenario.
    min_samples = np.min(samples, axis=0)
    
    # We want to maximize the score, so we negate the minimum sampled value
    scores = -min_samples
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
