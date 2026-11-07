import numpy as np
import time
from scipy.stats import norm

# Global timer for deadline guard
_start_time = time.time()
_DEADLINE = 330.0

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Michalewicz 2D.
    Uses Thompson Sampling for minimization.
    
    Parameters:
    mu (np.ndarray): Posterior mean, shape (2000,).
    sigma (np.ndarray): Posterior standard deviation, shape (2000,).
    f_best (float): Best observed value so far.
    
    Returns:
    np.ndarray: Scores for the 2000 candidates. Higher is better.
    """
    # Deadline guard
    if time.time() - _start_time > _DEADLINE - 5.0:
        # Fallback to pure exploitation to save time
        scores = -mu
        scores = np.where(np.isfinite(scores), scores, -1e6)
        return scores

    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-10)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-10)
    mu = np.where(np.isfinite(mu), mu, f_best)

    # Thompson Sampling: Sample from the posterior
    # We draw a single random vector to sample a function realization
    # y ~ N(mu, sigma^2)
    # We want to minimize y, so we return -y as the score (higher is better)
    
    # Generate standard normal random numbers
    eps = np.random.randn(mu.shape[0])
    
    # Compute sampled values
    sampled_values = mu + sigma * eps
    
    # Return negative of sampled values so that the minimum value gets the highest score
    scores = -sampled_values
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -1e6)
    
    return scores
