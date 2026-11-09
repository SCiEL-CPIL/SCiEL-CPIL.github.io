import numpy as np
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = 330.0  # seconds total budget, leave margin

# Use a deterministic seed for Thompson Sampling to ensure reproducibility
_TS_RNG = np.random.default_rng(42)

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Hartmann 6D.
    Uses Thompson Sampling: sample from the posterior Gaussian and return
    negative sampled values as scores (higher score = more promising).
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure finite inputs
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    
    # Ensure sigma is non-negative
    sigma = np.maximum(sigma, 0.0)
    
    # Check if we're approaching the deadline
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE - 10:
        # Fallback to a simple exploitation strategy if time is running out
        score = -mu
        return np.where(np.isfinite(score), score, 0.0)
    
    # Thompson Sampling: sample from N(mu, sigma^2) for each candidate
    # We want to minimize, so we sample the predicted value and select the minimum
    # Score = -sampled_value (higher score means lower predicted value)
    samples = _TS_RNG.normal(loc=mu, scale=sigma)
    
    # Score: negative of the sampled value (higher is better)
    score = -samples
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
