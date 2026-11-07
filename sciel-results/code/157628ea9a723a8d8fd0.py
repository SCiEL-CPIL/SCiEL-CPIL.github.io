"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition function.
    
    Generates a single sample from the posterior distribution:
    sample = mu - sigma * Z, where Z ~ N(0, 1)
    
    For minimization, we want to find the minimum of the sampled function.
    The acquisition score is the negative of the sampled value, so that
    higher scores correspond to lower predicted values.
    
    Args:
        mu: array of shape (2000,) - posterior mean
        sigma: array of shape (2000,) - posterior std
        f_best: float - best observed value so far
    
    Returns:
        array of shape (2000,) - acquisition scores (higher is better for selection)
    """
    # Ensure inputs are numpy arrays and finite
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Replace non-finite values in mu and sigma with safe defaults
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    
    # Ensure sigma is non-negative for safety
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Generate a single standard normal random vector
    Z = np.random.standard_normal(len(mu))
    
    # Sample from the posterior: sample = mu - sigma * Z
    # For minimization, we want to minimize this sample
    # The acquisition score is -sample = -mu + sigma * Z
    scores = -mu + sigma_safe * Z
    
    # Ensure the output is finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
