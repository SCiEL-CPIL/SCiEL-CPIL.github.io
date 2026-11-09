"""Acquisition function for Bayesian optimization of Beale 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining expected improvement with an uncertainty bonus.
    
    Parameters:
    -----------
    mu : np.ndarray, shape (2000,)
        Mean of the GP posterior at 2000 candidate points.
    sigma : np.ndarray, shape (2000,)
        Standard deviation of the GP posterior at 2000 candidate points.
    f_best : float
        Best (lowest) observed objective value so far.
    
    Returns:
    --------
    np.ndarray, shape (2000,)
        Acquisition scores; higher means more promising to evaluate next.
    """
    # Clip sigma to avoid division by zero and numerical issues
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Expected Improvement (EI)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Adaptive exploration weight: larger when f_best is far from optimum (0),
    # smaller when f_best is close. Beale minimum is 0, max is ~60.
    # Use a smooth decay: weight = base * (1 + f_best / scale) clipped
    # When f_best ~ 60 (early), weight ~ base * 3
    # When f_best ~ 0.1 (late), weight ~ base * 1.002
    # This gives strong early exploration, gentle late exploration
    f_best_clipped = max(f_best, 1e-6)
    scale = 10.0  # characteristic scale for Beale function values
    exploration_weight = 0.15 * (1.0 + f_best_clipped / scale)
    bonus = exploration_weight * sigma_safe
    
    return ei + bonus
# EVOLVE-BLOCK-END
