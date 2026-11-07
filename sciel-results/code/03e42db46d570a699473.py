"""Acquisition function for Bayesian optimization of sphere 1d."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    """
    Use Thompson Sampling for minimization.
    
    We sample a value from the posterior distribution of the objective function
    at each candidate point: f_hat = mu + sigma * z, where z ~ N(0, 1).
    We then select the point with the lowest sampled value.
    Since the evaluator selects the candidate with the highest acquisition score,
    we return the negative of the sampled value.
    """
    # Generate standard normal samples for each candidate
    z = np.random.randn(len(mu))
    
    # Compute sampled values
    # sigma is the standard deviation
    f_hat = mu + sigma * z
    
    # We want to minimize f, so we want the lowest f_hat.
    # The evaluator picks the max score.
    # Therefore, score = -f_hat
    scores = -f_hat
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
