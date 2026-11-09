"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Hybrid acquisition: blend of UCB and Expected Improvement.
    
    Uses a UCB term with adaptive beta and an EI term, weighted to balance
    exploration and exploitation. This helps avoid premature convergence in
    6D while still driving toward promising regions.
    
    Parameters:
    -----------
    mu : array_like, shape (2000,)
        Mean of the GP posterior at candidate points.
    sigma : array_like, shape (2000,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best (lowest) observed objective value so far.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher values are preferred.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-12)
    
    # UCB component: mu + beta * sigma
    # Use a moderate beta to encourage exploration in 6D
    beta = 1.5
    ucb = mu + beta * sigma
    
    # Expected Improvement component
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    z = (f_best - mu) / sigma
    # Clip z to avoid numerical issues
    z = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Normalize both components to have zero mean and unit variance
    # This makes the blend well-conditioned regardless of scale
    ucb_mean = np.mean(ucb)
    ucb_std = np.std(ucb)
    if ucb_std > 1e-12:
        ucb_norm = (ucb - ucb_mean) / ucb_std
    else:
        ucb_norm = np.zeros_like(ucb)
    
    ei_mean = np.mean(ei)
    ei_std = np.std(ei)
    if ei_std > 1e-12:
        ei_norm = (ei - ei_mean) / ei_std
    else:
        ei_norm = np.zeros_like(ei)
    
    # Blend: 50% UCB, 50% EI
    # This balances exploration (UCB) with exploitation (EI)
    scores = 0.5 * ucb_norm + 0.5 * ei_norm
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
