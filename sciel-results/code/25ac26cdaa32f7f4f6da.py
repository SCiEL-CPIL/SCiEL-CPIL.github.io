"""Acquisition function using Expected Improvement (EI)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Expected Improvement (EI) for minimization.
    
    EI(x) = E[max(0, f_best - f(x))]
    For a Gaussian posterior f(x) ~ N(mu, sigma^2):
    EI(x) = (f_best - mu) * Phi(z) + sigma * phi(z)
    where z = (f_best - mu) / sigma, Phi is CDF, phi is PDF of standard normal.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        Mean of the GP posterior at candidate points.
    sigma : array of shape (2000,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        The lowest observed value so far.
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores; higher values indicate more promising candidates.
    """
    # Ensure sigma is strictly positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Compute z-score
    z = (f_best - mu) / sigma_safe
    
    # Compute EI components
    # Phi(z) is the CDF of the standard normal distribution
    # phi(z) is the PDF of the standard normal distribution
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Note: If mu > f_best, (f_best - mu) is negative, but Phi(z) is small,
    # and the term sigma * phi(z) is positive.
    # Actually, the standard formula for minimization EI is:
    # Let d = f_best - mu.
    # If d > 0: EI = d * Phi(d/sigma) + sigma * phi(d/sigma)
    # If d <= 0: EI = 0 (since we expect no improvement if mean is already better)
    # However, the formula above works for all z if we just calculate it, 
    # but technically EI is 0 when mu >= f_best in many implementations to save compute,
    # but mathematically the integral is positive for any mu due to the tail, 
    # but standard EI definition for *improvement* over best is 0 if mean is better.
    # Let's stick to the standard analytic solution which is non-negative.
    
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # Clip to 0 to ensure non-negative scores (numerical stability)
    ei = np.maximum(ei, 0.0)
    
    # Ensure finite values
    scores = np.where(np.isfinite(ei), ei, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
