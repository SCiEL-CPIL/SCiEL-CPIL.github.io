"""Acquisition function for Bayesian optimization of branin 2d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Knowledge Gradient (KG) acquisition function for minimization.
    
    Parameters
    ----------
    mu : np.ndarray, shape (2000,)
        GP mean predictions at candidate points.
    sigma : np.ndarray, shape (2000,)
        GP standard deviation predictions at candidate points.
    f_best : float
        Best observed value so far (lowest).
    
    Returns
    -------
    scores : np.ndarray, shape (2000,)
        Finite scores; higher means evaluate next.
    """
    # Avoid division by zero and handle edge cases
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Z-score for improvement
    z = (f_best - mu) / sigma_safe
    
    # Standard normal CDF and PDF
    Phi = norm.cdf(z)
    phi = norm.pdf(z)
    
    # Knowledge Gradient approximation:
    # KG = E[max(f_best - Y, 0)] where Y ~ N(mu, sigma^2)
    # This is the expected improvement, but we can add a correction term
    # to better account for the global nature of the search.
    
    # Base EI (same as before)
    ei = (f_best - mu) * Phi + sigma_safe * phi
    ei = np.maximum(ei, 0.0)
    
    # KG correction: add a term that rewards exploration in regions
    # where there's significant uncertainty about whether the point
    # could be the new global minimum.
    # The idea: if sigma is large relative to (f_best - mu), there's
    # more chance this point could be the new best.
    
    # A practical KG approximation that adds exploration pressure
    # when uncertainty is high relative to the expected improvement
    uncertainty_ratio = sigma_safe / (np.abs(f_best - mu) + sigma_safe + 1e-9)
    
    # Weight the exploration bonus by how much uncertainty remains
    # relative to the current best
    exploration_bonus = 0.1 * uncertainty_ratio * sigma_safe
    
    scores = ei + exploration_bonus
    
    # Ensure all scores are finite and non-negative
    scores = np.where(np.isfinite(scores) & (scores >= 0), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
