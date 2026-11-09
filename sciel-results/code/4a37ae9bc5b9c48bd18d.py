"""Rewritten program using posterior sampling-based acquisition."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Posterior sampling-based acquisition function.
    
    Approximates Expected Improvement by drawing 50 latent function samples
    from the GP posterior and computing the expected improvement as the mean
    of the maximum improvement over f_best across these samples.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Mean of the GP posterior at candidate points.
    sigma : array-like, shape (2000,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best (lowest) observed value so far.
    
    Returns:
    --------
    scores : array, shape (2000,)
        Acquisition scores; higher values indicate more promising candidates.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    n_samples = 50
    n_points = mu.shape[0]
    
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Draw n_samples samples from the GP posterior at all candidate points
    # Each sample is a vector of length n_points
    # samples has shape (n_samples, n_points)
    z = np.random.standard_normal((n_samples, n_points))
    samples = mu[None, :] + sigma_safe[None, :] * z
    
    # For each sample, compute the improvement over f_best: max(0, f_best - sample)
    # improvements has shape (n_samples, n_points)
    improvements = np.maximum(0.0, f_best - samples)
    
    # The acquisition score for each point is the mean improvement across samples
    # This approximates E[max(0, f_best - f(x))]
    scores = np.mean(improvements, axis=0)
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
