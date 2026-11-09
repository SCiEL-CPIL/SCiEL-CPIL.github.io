"""Rewritten program using Expected Improvement (EI) acquisition function."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Expected Improvement (EI) acquisition function for minimization.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean of the GP posterior at candidate points (in objective units).
    sigma : ndarray, shape (2000,)
        Standard deviation of the GP posterior at candidate points (in objective units).
    f_best : float
        Best (lowest) observed function value so far.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher values are preferred for evaluation.
    """
    # Ensure inputs are 1D arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Clip sigma to avoid division by zero and numerical instability
    # Using a small epsilon to prevent log(0) or inf in Z calculation
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Calculate Z-score: (f_best - mu) / sigma
    # For minimization, we look for points where mu < f_best
    Z = (f_best - mu) / sigma_safe
    
    # Expected Improvement formula:
    # EI = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    # Note: When f_best - mu < 0, EI should be 0.
    # The formula naturally handles this because:
    # If Z is very negative, Phi(Z) ~ 0 and phi(Z) ~ 0.
    # However, to be strictly correct and avoid negative EI values due to floating point errors
    # or theoretical extensions, we can clip the result to be non-negative.
    
    # Calculate components
    phi_Z = norm.pdf(Z)
    Phi_Z = norm.cdf(Z)
    
    # EI calculation
    ei = (f_best - mu) * Phi_Z + sigma_safe * phi_Z
    
    # Ensure EI is non-negative
    # Mathematically EI >= 0. Numerical errors might cause tiny negative values.
    ei = np.maximum(ei, 0.0)
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(ei), ei, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
