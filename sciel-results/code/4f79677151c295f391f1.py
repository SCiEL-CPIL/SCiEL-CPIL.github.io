"""Acquisition function for Bayesian optimization of Hartmann 6D using Thompson Sampling."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition function.
    
    Samples one value from the posterior Gaussian for each candidate and returns
    negative sampled values as scores (higher score = lower sampled value = better).
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Posterior mean at each candidate point.
    sigma : array-like, shape (2000,)
        Posterior standard deviation at each candidate point.
    f_best : float
        Best (lowest) observed function value so far.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Finite scores where higher means the candidate should be evaluated next.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-12)
    
    # Sample from the posterior: y ~ N(mu, sigma^2)
    # Use a fixed random seed for determinism within a single call is not appropriate
    # since we want different samples each call. We just draw fresh random numbers.
    rng = np.random.default_rng()
    samples = mu + sigma * rng.standard_normal(mu.shape)
    
    # Thompson Sampling for minimization: pick the point with the lowest sampled value.
    # We return scores such that higher score = lower sampled value.
    # So we return negative of the samples.
    scores = -samples
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
