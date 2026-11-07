"""Acquisition function for Bayesian optimization of beale 2d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Entropy-based acquisition function for minimization.
    
    Uses the log of posterior variance as a proxy for information gain,
    which encourages exploration of high-uncertainty regions to map the
    multimodal landscape of the Beale function more efficiently.
    
    Parameters
    ----------
    mu : array_like, shape (n,)
        Posterior mean at candidate points.
    sigma : array_like, shape (n,)
        Posterior standard deviation at candidate points.
    f_best : float
        Best (lowest) observed function value so far.

    Returns
    -------
    scores : ndarray, shape (n,)
        Acquisition scores; higher values indicate more promising candidates.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Guard against non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    sigma = np.maximum(sigma, 1e-12)
    
    # Entropy-based scoring: log(sigma^2) = 2 * log(sigma)
    # This directly targets regions of highest posterior uncertainty
    scores = 2.0 * np.log(sigma)
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
