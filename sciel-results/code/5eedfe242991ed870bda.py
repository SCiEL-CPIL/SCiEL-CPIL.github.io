"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function that blends Expected Improvement (EI) and
    Lower Confidence Bound (LCB) to balance exploitation and exploration.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean of the Gaussian process posterior.
    sigma : ndarray, shape (2000,)
        Standard deviation of the Gaussian process posterior.
    f_best : float
        Best (lowest) observed function value so far.
        
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means more promising.
    """
    # Ensure sigma is positive to avoid division by zero
    sigma = np.maximum(sigma, 1e-9)
    
    # Expected Improvement
    z_ei = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z_ei) + sigma * norm.pdf(z_ei)
    
    # Lower Confidence Bound (exploration)
    # We want to minimize, so LCB = mu - beta * sigma
    # Higher LCB score means more exploratory
    beta = 0.5  # Exploration parameter
    lcb = mu - beta * sigma
    
    # Normalize both to [0, 1] range for blending
    # EI is always >= 0, so normalize by its max
    ei_max = np.max(ei) if np.max(ei) > 0 else 1.0
    ei_norm = ei / ei_max
    
    # LCB can be negative; normalize by range
    lcb_min = np.min(lcb)
    lcb_max = np.max(lcb)
    lcb_range = lcb_max - lcb_min if lcb_max > lcb_min else 1.0
    lcb_norm = (lcb - lcb_min) / lcb_range
    
    # Blend: 70% EI, 30% LCB
    scores = 0.7 * ei_norm + 0.3 * lcb_norm
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
