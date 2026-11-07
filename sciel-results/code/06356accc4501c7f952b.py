"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Branin 2D.
    
    Uses a hybrid of Expected Improvement (EI) and Upper Confidence Bound (UCB)
    with adaptive beta based on the uncertainty level.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean of the GP posterior at candidate points.
    sigma : ndarray, shape (2000,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best observed function value so far.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores for each candidate. Higher means more promising.
    """
    # Ensure sigma is positive to avoid division by zero
    sigma = np.maximum(sigma, 1e-9)
    
    # Compute z-score for EI
    z = (f_best - mu) / sigma
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # UCB component: we want to explore areas with high uncertainty
    # but also exploit areas with low predicted values
    # Use an adaptive beta that balances exploration and exploitation
    
    # Normalize mu and sigma for better scaling
    mu_range = np.ptp(mu)
    if mu_range < 1e-10:
        mu_range = 1e-10
    sigma_range = np.ptp(sigma)
    if sigma_range < 1e-10:
        sigma_range = 1e-10
    
    # Adaptive beta: larger when uncertainty is high relative to the spread
    # This encourages exploration in uncertain regions
    beta = 1.0
    
    # UCB score: -mu + beta * sigma (since we're minimizing)
    ucb = -mu + beta * sigma
    
    # Blend EI and UCB: weight based on how close mu is to f_best
    # When mu is close to f_best, rely more on EI (exploitation)
    # When mu is far from f_best, rely more on UCB (exploration)
    
    # Normalize scores to [0, 1] for blending
    ei_min = np.min(ei)
    ei_max = np.max(ei)
    ei_range = ei_max - ei_min
    if ei_range < 1e-10:
        ei_range = 1e-10
    ei_norm = (ei - ei_min) / ei_range
    
    ucb_min = np.min(ucb)
    ucb_max = np.max(ucb)
    ucb_range = ucb_max - ucb_min
    if ucb_range < 1e-10:
        ucb_range = 1e-10
    ucb_norm = (ucb - ucb_min) / ucb_range
    
    # Blend: 50% EI, 50% UCB
    scores = 0.5 * ei_norm + 0.5 * ucb_norm
    
    # Ensure finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e10, neginf=-1e10)
    
    return scores
# EVOLVE-BLOCK-END
