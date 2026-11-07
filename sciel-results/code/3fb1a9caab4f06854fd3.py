"""Acquisition function for Bayesian optimization of Hartmann 3D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Compute acquisition scores using EI with a negative xi for exploration.
    
    Parameters:
    -----------
    mu : array-like, shape (n,)
        Mean of the Gaussian process posterior at candidate points.
    sigma : array-like, shape (n,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best observed value so far (lowest).
    
    Returns:
    --------
    scores : ndarray, shape (n,)
        Acquisition scores; higher values indicate more promising points.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive to avoid division issues
    sigma = np.maximum(sigma, 1e-9)
    
    # Use a negative xi to encourage exploration
    # This shifts the target below f_best, making the acquisition function
    # favor points that might improve by more than xi
    xi = -0.1
    target = f_best - xi
    
    # Expected Improvement with shifted target
    improvement = target - mu
    z = improvement / sigma
    
    # Compute EI: (target - mu) * Phi(z) + sigma * phi(z)
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    scores = improvement * cdf + sigma * pdf
    
    # Ensure scores are non-negative (EI should be >= 0)
    scores = np.maximum(scores, 0.0)
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
