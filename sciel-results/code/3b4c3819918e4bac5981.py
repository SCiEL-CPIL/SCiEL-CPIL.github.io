"""Rewritten acquisition for Bayesian optimization of Beale 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Gaussian EI with adaptive exploration.
    
    Combines standard Gaussian Expected Improvement with an uncertainty bonus
    that emphasizes exploration in regions far from current best, while
    exploiting near the basin once it's found.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Posterior mean at candidate points.
    sigma : ndarray, shape (2000,)
        Posterior standard deviation at candidate points.
    f_best : float
        Best observed function value so far.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means more promising.
    """
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Gaussian Expected Improvement
    z = (f_best - mu) / sigma_safe
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    ei = np.clip(ei, -1e6, 1e6)
    
    # Adaptive exploration: weight uncertainty by how promising the region looks
    # When mu is close to f_best, reduce exploration to focus on exploitation
    # When mu is far from f_best, increase exploration to find the basin
    mu_distance = np.abs(mu - f_best)
    
    # Exploration coefficient that decreases as we get closer to f_best
    # This helps focus search once we're near the minimum
    explore_weight = 0.15 + 0.1 * np.exp(-mu_distance / max(f_best, 1.0))
    
    scores = ei + explore_weight * sigma_safe
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
