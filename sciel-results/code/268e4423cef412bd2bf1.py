import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Computes the Thompson Sampling acquisition function for minimization.
    Draws a single scalar sample z from N(0,1) and computes the posterior
    predictive sample at each candidate as mu - sigma * z. Returns the negative
    of this sample as the score (higher is better for minimization).
    
    Parameters:
    mu : array-like, shape (N,)
        Posterior mean at N candidate points.
    sigma : array-like, shape (N,)
        Posterior standard deviation at N candidate points.
    f_best : float
        Current best observed function value.
        
    Returns:
    scores : ndarray, shape (N,)
        Thompson Sampling scores for the candidate points (higher is better).
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is strictly positive
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Thompson Sampling: Draw a single scalar sample z from N(0,1)
    # and compute the "sampled" function value at each candidate.
    # For minimization, we want to find the candidate with the lowest sampled value.
    # So we return the negative of the sampled value as the score.
    z = np.random.normal(0.0, 1.0)
    
    # Sampled value: f_sampled = mu + sigma * z
    # Score: -f_sampled = -mu - sigma * z
    scores = -mu - sigma_safe * z
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
