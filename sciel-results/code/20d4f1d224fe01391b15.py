"""Rewritten acquisition for Bayesian optimization of Hartmann 3D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function: a structural blend that combines a strong exploitation
    signal (negative mean) with a calibrated uncertainty term, plus a small
    Expected Improvement component. This differs from pure EI by explicitly
    rewarding low predicted mean values, which is critical for functions with
    multiple local minima like Hartmann 3D.

    Parameters
    ----------
    mu : ndarray, shape (n,)
        GP mean predictions at candidate points.
    sigma : ndarray, shape (n,)
        GP standard deviation predictions at candidate points.
    f_best : float
        Best (lowest) observed objective value so far.

    Returns
    -------
    scores : ndarray, shape (n,)
        Acquisition scores; higher means more promising.
    """
    # Ensure numerical stability
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standardized improvement for EI
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Structural change: combine three signals with different weights
    # 1. Exploitation: strongly reward low predicted mean (negative mu)
    # 2. Exploration: moderate reward for high uncertainty
    # 3. EI: standard improvement-based score
    
    # Normalize each component to have similar scale
    mu_min = np.min(mu)
    mu_max = np.max(mu)
    mu_range = mu_max - mu_min if mu_max > mu_min else 1.0
    
    # Exploitation score: lower mu is better, so score = -mu normalized
    exploit_score = -(mu - mu_min) / mu_range  # in [0, 1], higher is better
    
    # Exploration score: higher sigma is better
    sigma_min = np.min(sigma_safe)
    sigma_max = np.max(sigma_safe)
    sigma_range = sigma_max - sigma_min if sigma_max > sigma_min else 1.0
    explore_score = (sigma_safe - sigma_min) / sigma_range  # in [0, 1]
    
    # EI score: normalize to [0, 1]
    ei_max = np.max(ei)
    ei_min = np.min(ei)
    ei_range = ei_max - ei_min if ei_max > ei_min else 1.0
    ei_score = (ei - ei_min) / ei_range  # in [0, 1]
    
    # Weighted combination: emphasize exploitation, balance exploration
    w_exploit = 0.6
    w_explore = 0.2
    w_ei = 0.2
    
    scores = w_exploit * exploit_score + w_explore * explore_score + w_ei * ei_score
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
