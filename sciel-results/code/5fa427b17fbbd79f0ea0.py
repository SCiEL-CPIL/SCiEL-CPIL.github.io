"""Rewritten acquisition function for Bayesian optimization of Styblinski-Tang 1D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of styblinski tang 1d.
    
    Uses Pure Expected Improvement (EI).
    This is a structurally simpler and theoretically grounded approach
    compared to blended heuristics. EI directly maximizes the expected
    reduction in the objective function value.
    
    Parameters:
    -----------
    mu : array_like, shape (2000,)
        Mean of the GP posterior at candidate points.
    sigma : array_like, shape (2000,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best observed value so far (minimum).
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores for each candidate. Higher is better.
    """
    start_time = time.time()
    deadline = start_time + 320.0  # Safety margin before 330s limit
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive to avoid division by zero
    sigma = np.maximum(sigma, 1e-9)
    
    # Calculate Expected Improvement for minimization
    # EI(x) = (f_best - mu(x)) * Phi(z) + sigma(x) * phi(z)
    # where z = (f_best - mu(x)) / sigma(x)
    z = (f_best - mu) / sigma
    
    # Use stable calculation for norm.cdf and norm.pdf
    # For very negative z, cdf is 0, pdf is 0.
    # For very positive z, cdf is 1, pdf is 0.
    
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_z + sigma * pdf_z
    
    # EI is non-negative. Higher is better.
    # Ensure finite values
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # Check deadline
    if time.time() > deadline:
        return ei
    
    return ei
# EVOLVE-BLOCK-END
