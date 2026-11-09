import numpy as np
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = 330.0  # seconds total budget, leave margin

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Hartmann 6D.
    Uses Thompson Sampling: sample from posterior N(mu, sigma^2) for each candidate
    and return the negative sampled value as the score (higher = more promising).
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure finite inputs
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    
    # Ensure sigma is non-negative
    sigma = np.maximum(sigma, 0.0)
    
    # Thompson Sampling: draw one sample from the posterior for each candidate
    # Use a fixed random seed per call? No, we want randomness each time.
    # But we need to be careful: the evaluator calls acquisition once per step.
    # Each call should use fresh random numbers.
    
    # Sample from N(mu, sigma^2)
    # Use standard normal and scale
    n = mu.shape[0]
    # Generate standard normal samples
    z = np.random.standard_normal(n)
    samples = mu + sigma * z
    
    # For minimization, we want the candidate with the lowest sampled value.
    # Score should be higher for more promising candidates.
    # So score = -samples (negate so lower sampled value gives higher score)
    score = -samples
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
