import numpy as np
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = 330.0  # seconds total budget, leave margin

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Hartmann 6D.
    Uses Thompson Sampling: sample from posterior, return negative sample
    so that the minimum sampled value gets the highest score.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure finite inputs
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    
    # Avoid zero/negative sigma
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    
    # Thompson Sampling: draw one sample from N(mu, sigma^2) for each candidate
    # Use standard normal random numbers
    z = np.random.standard_normal(len(mu))
    samples = mu + sigma_safe * z
    
    # For minimization, we want to pick the candidate with the lowest sampled value.
    # Return negative samples so that higher score = better (lower sampled value).
    score = -samples
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, -1e10)
    
    return score
