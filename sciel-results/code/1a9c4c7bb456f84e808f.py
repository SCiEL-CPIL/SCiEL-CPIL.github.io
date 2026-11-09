"""Acquisition function for Bayesian optimization of Hartmann 3D."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# We use a hybrid acquisition: a combination of Expected Improvement (EI)
# and a penalty term that encourages exploration in regions with high uncertainty
# but also balances exploitation. We use a simple, fast approach.

# Precompute constants for speed
_EPS = 1e-9


def _norm_cdf(z):
    """Vectorized normal CDF."""
    return 0.5 * (1.0 + np.erf(z / np.sqrt(2.0)))


def _norm_pdf(z):
    """Vectorized normal PDF."""
    return np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)


def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining Expected Improvement with an exploration bonus.
    
    Parameters
    ----------
    mu : array-like, shape (2000,)
        Posterior mean predictions.
    sigma : array-like, shape (2000,)
        Posterior standard deviation predictions.
    f_best : float
        Best observed value so far.
    
    Returns
    -------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher is better).
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, _EPS)
    
    # Compute Expected Improvement
    improvement = f_best - mu
    z = improvement / sigma
    ei = improvement * _norm_cdf(z) + sigma * _norm_pdf(z)
    
    # Add a small exploration bonus based on uncertainty
    # This helps escape local optima in the acquisition landscape
    # The bonus is scaled relative to the typical EI values
    # We use a small coefficient to not dominate the EI term
    exploration_bonus = 0.01 * sigma
    
    scores = ei + exploration_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
