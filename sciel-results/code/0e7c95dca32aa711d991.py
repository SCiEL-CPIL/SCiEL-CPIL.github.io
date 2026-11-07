"""Acquisition function for Bayesian optimization of Michalewicz 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard: kill after 330s wall-clock, use safety margin of 10s
_DEADLINE_START = None
_DEADLINE_LIMIT = 330.0 - 10.0  # 320s safety margin


def _get_remaining_time():
    global _DEADLINE_START
    if _DEADLINE_START is None:
        _DEADLINE_START = time.time()
    return max(0.0, _DEADLINE_LIMIT - (time.time() - _DEADLINE_START))


def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling with exploitation bias for Michalewicz 2D minimization.
    
    mu, sigma: shape (2000,), posterior mean and std in objective units
    f_best: lowest observed value
    Returns: 2000 finite scores, higher = more likely to be selected
    """
    n = len(mu)
    
    # Thompson Sampling: sample from posterior predictive
    # For minimization, we want to find low values
    # Sample: theta = mu - sigma * z where z ~ N(0,1)
    # This gives us a sample of the true function value at each candidate
    
    # Use a fixed seed per call is not possible since we don't have state,
    # but we can use a deterministic pseudo-random based on the data itself
    # Actually, for Thompson Sampling in BO, we should use random samples.
    # The evaluator calls this function repeatedly, so we need randomness.
    
    # Generate standard normal samples
    z = np.random.standard_normal(n)
    
    # Thompson sample: lower is better, so we score by the negative sample
    # Sample value: s = mu - sigma * z (this is a sample from N(mu, sigma^2))
    # We want to maximize the score, so we use -s as the score
    ts_scores = -(mu - sigma * z)
    
    # Add exploitation bias: prefer points with low mu (known good regions)
    # This helps escape local minima while still exploiting known good areas
    # Use a small deterministic bias toward low mu
    exploitation_bias = 0.1 * (-mu)  # negative mu gets positive bias
    
    # Combine Thompson Sampling score with exploitation bias
    scores = ts_scores + exploitation_bias
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
