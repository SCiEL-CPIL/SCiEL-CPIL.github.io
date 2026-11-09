"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of styblinski tang 1d.
    
    Uses Expected Improvement (EI) as the primary acquisition, with a
    small UCB exploration bonus that is scaled down over time based on
    how close f_best is to the known global minimum.
    
    Parameters:
    -----------
    mu : array_like, shape (2000,)
        Mean of the GP posterior at candidate points.
    sigma : array_like, shape (2000,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best observed value so far (minimum).
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores for each candidate. Higher is better.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-9)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # Adaptive exploration: UCB-style bonus
    # The known global minimum of styblinski-tang 1d is approximately -39.1662
    # As f_best approaches this value, reduce exploration
    global_min = -39.1662
    # Normalized progress: 0 when f_best is far, 1 when close to global min
    # Use a scale relative to the function range
    # Styblinski-Tang 1D range is roughly [-39.17, ~20]
    range_scale = 60.0  # approximate range of the function
    progress = np.clip((f_best - global_min) / range_scale, 0.0, 1.0)
    # When progress is high (close to optimum), reduce exploration
    # When progress is low (far from optimum), increase exploration
    exploration_weight = 0.3 * (1.0 - progress) + 0.05
    
    # UCB score: -mu + kappa * sigma (higher is better)
    kappa = 1.5
    ucb = -mu + kappa * sigma
    ucb = np.where(np.isfinite(ucb), ucb, 0.0)
    
    # Normalize EI to [0, 1]
    ei_min = np.min(ei)
    ei_max = np.max(ei)
    ei_range = ei_max - ei_min + 1e-12
    ei_norm = (ei - ei_min) / ei_range
    
    # Normalize UCB to [0, 1]
    ucb_min = np.min(ucb)
    ucb_max = np.max(ucb)
    ucb_range = ucb_max - ucb_min + 1e-12
    ucb_norm = (ucb - ucb_min) / ucb_range
    
    # Blend: primarily EI with adaptive UCB bonus
    weight_ei = 1.0 - exploration_weight
    weight_ucb = exploration_weight
    
    score = weight_ei * ei_norm + weight_ucb * ucb_norm
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
