"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function that blends Expected Improvement (EI) with a
    lower-confidence-bound (LCB) style term to encourage exploration
    in high-uncertainty regions, while still exploiting promising areas.

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
    # Ensure finite input; replace non-finite with safe defaults
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Guard against non-finite values
    mu = np.where(np.isfinite(mu), mu, np.nanmax(mu) if np.any(np.isfinite(mu)) else 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    sigma = np.maximum(sigma, 1e-12)  # avoid division by zero

    # Expected Improvement
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # LCB-style exploration term: favor points with low mean and high uncertainty
    # We use a tunable beta; here we set it adaptively based on the scale of sigma
    # A fixed beta works well for this problem; we choose beta = 1.5
    lcb = -mu + 1.5 * sigma
    
    # Blend: primarily EI, but add a small exploration boost
    # Normalize both to comparable scales
    ei_min, ei_max = np.min(ei), np.max(ei)
    lcb_min, lcb_max = np.min(lcb), np.max(lcb)
    
    ei_range = ei_max - ei_min if ei_max > ei_min else 1.0
    lcb_range = lcb_max - lcb_min if lcb_max > lcb_min else 1.0
    
    ei_norm = (ei - ei_min) / ei_range
    lcb_norm = (lcb - lcb_min) / lcb_range
    
    # Weighted combination: 80% EI, 20% LCB
    scores = 0.8 * ei_norm + 0.2 * lcb_norm
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
