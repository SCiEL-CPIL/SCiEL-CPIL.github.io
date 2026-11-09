"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of branin 2d.
    
    Uses Thompson Sampling: draws a single sample from the posterior
    at each candidate point and returns the negative sample as the score.
    Maximizing this score is equivalent to minimizing the sampled function value.
    
    Args:
        mu: array of shape (2000,) with posterior means in objective units
        sigma: array of shape (2000,) with posterior standard deviations in objective units
        f_best: the lowest observed value so far
    
    Returns:
        array of shape (2000,) with acquisition scores (higher = more promising)
    """
    # Ensure sigma is non-negative and handle any numerical issues
    sigma = np.maximum(sigma, 1e-9)
    
    # Generate standard normal random numbers for each candidate
    # Shape must match mu and sigma
    n = mu.shape[0]
    eps = np.random.randn(n)
    
    # Draw samples from the posterior: f ~ N(mu, sigma^2)
    samples = mu + sigma * eps
    
    # We want to minimize f, so we maximize -f
    scores = -samples
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e10, neginf=-1e10)
    
    return scores
# EVOLVE-BLOCK-END
