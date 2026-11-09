"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Branin 2D.
    
    Uses standard Expected Improvement (EI) for minimization.
    This approach was identified as optimal in successful inspirations, 
    balancing exploration and exploitation via the GP posterior uncertainty 
    without arbitrary heuristic bonuses.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Mean of the GP posterior at candidate points.
    sigma : array-like, shape (2000,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best observed function value so far.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher is better.
    """
    start_time = time.time()
    # Safety margin: ensure we return within the 330s limit if called repeatedly or in a loop
    # Though this function is vectorized and fast, we guard against pathological inputs.
    deadline = 330.0 - 5.0
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive and finite to avoid division by zero and NaNs
    sigma = np.maximum(sigma, 1e-9)
    
    # Check for time violation (unlikely for vectorized ops, but good practice per directive)
    if time.time() > deadline:
        # Fallback: return random or zero scores if we are stuck (should not happen)
        return np.zeros_like(mu)
    
    # Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    
    # Compute standard normal CDF and PDF
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_z + sigma * pdf_z
    
    # Handle numerical issues: EI should be non-negative, but floating point errors 
    # might cause tiny negative values. Clip to 0.
    ei = np.maximum(ei, 0.0)
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(ei), ei, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
