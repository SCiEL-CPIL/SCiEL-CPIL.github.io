"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_start_time = time.time()
_DEADLINE = 325.0


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Weierstrass 1D.
    Uses Expected Improvement (EI) combined with Upper Confidence Bound (UCB)
    to balance exploitation and exploration for the multimodal landscape.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Mean of the Gaussian process posterior.
    sigma : array-like, shape (2000,)
        Standard deviation of the Gaussian process posterior.
    f_best : float
        Best observed value so far (minimum).
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores, higher means more promising.
    """
    # Deadline guard
    if time.time() - _start_time > _DEADLINE:
        return -mu + 2.0 * np.maximum(sigma, 1e-9)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement (EI):
    # Measures how much lower f(x) is expected to be compared to f_best.
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Upper Confidence Bound (UCB):
    # Balances exploitation (low mu) with exploration (high sigma).
    # beta controls the exploration-exploitation tradeoff.
    # For Weierstrass 1D (multimodal), we want moderate exploration.
    beta = 1.5
    ucb = -mu + beta * sigma_safe
    
    # Normalize both to [0, 1] for fair combination
    ei_min, ei_max = np.min(ei), np.max(ei)
    ei_range = ei_max - ei_min
    if ei_range > 1e-12:
        ei_norm = (ei - ei_min) / ei_range
    else:
        ei_norm = np.zeros_like(ei)
    
    ucb_min, ucb_max = np.min(ucb), np.max(ucb)
    ucb_range = ucb_max - ucb_min
    if ucb_range > 1e-12:
        ucb_norm = (ucb - ucb_min) / ucb_range
    else:
        ucb_norm = np.zeros_like(ucb)
    
    # Combine EI and UCB with weights favoring exploitation
    # but maintaining enough exploration for multimodal search
    w_ei = 0.6
    w_ucb = 0.4
    
    scores = w_ei * ei_norm + w_ucb * ucb_norm
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
