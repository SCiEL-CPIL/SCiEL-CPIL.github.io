"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global timing guard
_ACQ_START = time.time()
_TIME_LIMIT = 310.0  # Safety margin before 330s wall-clock limit


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of branin 2d.
    
    Uses a blend of Expected Improvement (EI) and UCB for robust exploration/exploitation.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Posterior mean at candidate points.
    sigma : array-like, shape (2000,)
        Posterior standard deviation at candidate points.
    f_best : float
        Best observed objective value so far (lower is better).
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means more likely to be evaluated next.
    """
    # Check time budget
    elapsed = time.time() - _ACQ_START
    if elapsed > _TIME_LIMIT:
        # Return a safe fallback: pick the point with lowest mu
        mu_arr = np.asarray(mu)
        return -mu_arr

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive to avoid division issues
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # UCB for minimization: mu - beta * sigma
    # We want to maximize this score, so we use - (mu - beta * sigma) = -mu + beta * sigma
    beta = 2.0
    ucb_score = -mu + beta * sigma_safe
    
    # Normalize EI and UCB to [0, 1] to combine them effectively
    ei_min = np.min(ei)
    ei_max = np.max(ei)
    if ei_max - ei_min > eps:
        ei_norm = (ei - ei_min) / (ei_max - ei_min)
    else:
        ei_norm = np.zeros_like(ei)
        
    ucb_min = np.min(ucb_score)
    ucb_max = np.max(ucb_score)
    if ucb_max - ucb_min > eps:
        ucb_norm = (ucb_score - ucb_min) / (ucb_max - ucb_min)
    else:
        ucb_norm = np.zeros_like(ucb_score)
        
    # Combine: 70% EI, 30% UCB
    scores = 0.7 * ei_norm + 0.3 * ucb_norm
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
