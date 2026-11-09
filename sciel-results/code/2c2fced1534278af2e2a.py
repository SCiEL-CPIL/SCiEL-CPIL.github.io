import numpy as np
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = 330.0  # seconds total budget, leave margin

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Hartmann 6D.
    Uses Thompson Sampling: sample from posterior N(mu, sigma^2) for each candidate.
    Return negative sampled values as scores (higher score = lower sampled value).
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure finite inputs
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    
    # Clip sigma to avoid numerical issues
    sigma = np.clip(sigma, 1e-10, 1e6)
    
    # Thompson Sampling: draw one sample from N(mu, sigma^2) for each candidate
    # Use a fixed random seed per call based on f_best to ensure reproducibility within a run
    # but different samples across calls
    rng = np.random.RandomState(int(abs(hash(f_best)) % (2**32)))
    
    # Sample from the posterior: sample = mu + sigma * z, where z ~ N(0,1)
    z = rng.standard_normal(size=mu.shape)
    samples = mu + sigma * z
    
    # Ensure samples are finite
    samples = np.where(np.isfinite(samples), samples, mu)
    
    # For minimization, we want the candidate with the lowest sampled value to have the highest score
    # So score = -sample (higher score means lower sampled value)
    score = -samples
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
