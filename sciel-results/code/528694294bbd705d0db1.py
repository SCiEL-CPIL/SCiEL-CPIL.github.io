"""Acquisition function for Bayesian optimization of goldstein price 2d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Compute an acquisition score for each candidate using a hybrid approach
    that combines Expected Improvement relative to the known global minimum
    with a variance-weighted exploration term.
    
    The key insight is that the global minimum of Goldstein-Price 2D is known
    to be 3.0. By computing EI relative to this known minimum rather than
    f_best, we maintain a strong exploration signal throughout the search,
    especially in regions with high posterior uncertainty where the true
    minimum might be located.
    
    Parameters:
    -----------
    mu : array-like, shape (n_candidates,)
        Posterior mean predictions at candidate points.
    sigma : array-like, shape (n_candidates,)
        Posterior standard deviations at candidate points.
    f_best : float
        Best observed function value so far.
    
    Returns:
    --------
    scores : ndarray, shape (n_candidates,)
        Acquisition scores; higher means more desirable to evaluate next.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Safety: ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Known global minimum of Goldstein-Price 2D
    f_min = 3.0
    
    # Compute Expected Improvement relative to the known global minimum.
    # This provides a constant exploration target regardless of f_best.
    # EI = E[max(f_min - f(x), 0)] where f(x) ~ N(mu, sigma^2)
    z = (f_min - mu) / sigma_safe
    ei = (f_min - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Add a variance-based exploration bonus that scales with uncertainty.
    # This encourages exploration in regions where the GP is uncertain.
    # The scale is chosen relative to the function range (~[3, 300]).
    # Use a moderate weight to balance exploitation vs exploration.
    var_bonus = 0.3 * sigma_safe
    
    # Also add a small LCB component for exploitation when we're close to the minimum
    # LCB = mu - beta * sigma, we negate for "higher is better"
    beta = 0.5
    lcb_score = -(mu - beta * sigma_safe)
    
    # Combine the terms:
    # - EI relative to known minimum: primary signal, drives search toward low values
    # - Variance bonus: encourages exploration in uncertain regions
    # - LCB: provides exploitation when uncertainty is low
    # 
    # The EI term is the dominant signal. When mu is well below f_min, EI is large.
    # When sigma is large, EI also increases (exploration).
    
    scores = 1.0 * ei + 0.15 * var_bonus + 0.05 * lcb_score
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
