"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining normalized EI with inverted UCB for
    exploration-exploitation balance in multi-modal optimization.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Mean of the GP posterior at candidate points.
    sigma : array-like, shape (2000,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best observed value so far.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher values indicate more promising candidates.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-9)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    
    # Compute normalized Expected Improvement
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Normalize EI to [0, 1] range for combination
    ei_max = np.max(ei) if np.max(ei) > 0 else 1.0
    ei_min = np.min(ei)
    ei_range = ei_max - ei_min
    if ei_range > 1e-12:
        ei_norm = (ei - ei_min) / ei_range
    else:
        ei_norm = np.zeros_like(ei)
    
    # Compute inverted UCB (uncertainty bonus)
    # Use a moderate exploration coefficient
    beta = 2.0
    ucb = mu + beta * sigma  # For minimization, we want to explore where mu is low and sigma is high
    # Invert: higher score means lower ucb value
    ucb_max = np.max(ucb)
    ucb_min = np.min(ucb)
    ucb_range = ucb_max - ucb_min
    if ucb_range > 1e-12:
        ucb_norm = (ucb_max - ucb) / ucb_range  # Invert so higher is better
    else:
        ucb_norm = np.zeros_like(ucb)
    
    # Combine: 70% exploitation (EI) + 30% exploration (inverted UCB)
    # This balance helps escape local optima in multi-modal functions
    w_ei = 0.7
    w_ucb = 0.3
    scores = w_ei * ei_norm + w_ucb * ucb_norm
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
