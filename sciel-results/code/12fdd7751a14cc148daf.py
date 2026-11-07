"""Rewritten program: Thompson Sampling acquisition for Michalewicz 2D BO."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition function.
    
    We approximate Thompson sampling by drawing a single Gaussian sample from the
    posterior at each candidate point, then returning the negative of that sample
    as the score (higher score = more promising for minimization).
    
    Parameters:
        mu: array of shape (2000,), posterior mean in objective units
        sigma: array of shape (2000,), posterior std in objective units
        f_best: float, lowest observed value so far
    
    Returns:
        scores: array of shape (2000,), finite, higher means evaluate next
    """
    # Ensure sigma is strictly positive to avoid numerical issues
    sigma = np.maximum(sigma, 1e-12)
    
    # Thompson Sampling: draw one sample from N(mu, sigma^2) at each point
    # Using a fixed random seed per call would make it deterministic but that's
    # not standard. Instead, we use the global numpy random state.
    # To make this reproducible within a single BO run, we rely on the fact
    # that the evaluator calls this in sequence; the randomness provides the
    # exploration inherent to Thompson Sampling.
    
    # Draw a single standard normal sample vector
    z = np.random.standard_normal(len(mu))
    
    # Posterior sample at each candidate
    samples = mu + sigma * z
    
    # For minimization, we want to select the point with the lowest sampled value.
    # Score = -sampled_value, so higher score corresponds to lower sampled value.
    scores = -samples
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
