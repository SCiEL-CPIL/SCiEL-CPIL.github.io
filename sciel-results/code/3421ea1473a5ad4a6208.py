# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of hartmann 3d.
    Uses Expected Improvement (EI) with a small UCB exploration term.
    
    Args:
        mu: array of shape (2000,), posterior mean at candidate points
        sigma: array of shape (2000,), posterior standard deviation at candidate points
        f_best: lowest observed value so far
    
    Returns:
        array of shape (2000,) with scores, higher meaning evaluate next
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Robust sigma floor to avoid division by zero and extreme values
    sigma_floor = 1e-8
    sigma_safe = np.maximum(sigma, sigma_floor)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    z_clipped = np.clip(z, -10.0, 10.0)
    
    # Expected Improvement
    cdf_val = norm.cdf(z_clipped)
    pdf_val = norm.pdf(z_clipped)
    
    ei = (f_best - mu) * cdf_val + sigma_safe * pdf_val
    ei = np.maximum(ei, 0.0)
    
    # Add a small UCB exploration term to encourage exploring high-uncertainty regions
    # This helps escape local optima in the surrogate landscape
    beta = 0.1
    ucb_term = beta * sigma_safe
    
    score = ei + ucb_term
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
