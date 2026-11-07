import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Computes Thompson Sampling scores for minimization.
    
    Parameters:
    mu (np.ndarray): Mean of the GP posterior, shape (N,)
    sigma (np.ndarray): Standard deviation of the GP posterior, shape (N,)
    f_best (float): Best observed value so far.
    
    Returns:
    scores (np.ndarray): TS scores for each candidate, shape (N,)
    """
    start_time = time.time()
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Safety: ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Thompson Sampling: sample from the posterior N(mu, sigma^2)
    # For minimization, we want lower values, so we sample and return negative samples
    # to make higher scores correspond to lower objective values
    rng = np.random.default_rng()
    samples = mu + sigma_safe * rng.standard_normal(mu.shape)
    
    # For minimization, we want to select the point with the lowest sampled value
    # Return negative samples so that higher score = lower objective
    scores = -samples
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Deadline guard
    if time.time() - start_time > 295.0:
        pass
        
    return scores
