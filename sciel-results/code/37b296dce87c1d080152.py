"""Acquisition function for Bayesian optimization of branin 2d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE_MARGIN = 5.0  # seconds before the 330s limit


def _check_deadline():
    """Return True if we should stop to avoid timeout."""
    return (time.time() - _start_time) > (330.0 - _DEADLINE_MARGIN)


def acquisition(mu, sigma, f_best):
    """
    Knowledge Gradient (KG) inspired acquisition function for minimization.
    
    Combines the expected improvement (EI) with a principled exploration bonus
    that accounts for the probability that a candidate could be the new global
    minimum. This provides a more robust exploration-exploitation balance than
    pure EI or PoI for multi-modal landscapes.
    
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
    # Deadline guard: if we're running out of time, return a safe fallback
    if _check_deadline():
        sigma_safe = np.maximum(sigma, 1e-8)
        z = (f_best - mu) / sigma_safe
        # Fallback to simple PoI for speed
        poi = norm.cdf(z)
        return np.where(np.isfinite(poi), poi, 0.0)
    
    # Avoid division by zero and handle edge cases
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Z-score for improvement
    z = (f_best - mu) / sigma_safe
    
    # Standard normal CDF and PDF
    Phi = norm.cdf(z)
    phi = norm.pdf(z)
    
    # Expected Improvement (EI) component
    ei = (f_best - mu) * Phi + sigma_safe * phi
    ei = np.maximum(ei, 0.0)
    
    # Knowledge Gradient correction:
    # The KG estimates the expected reduction in the best observed value.
    # A practical approximation adds an exploration bonus that is proportional
    # to the probability that this point is the new global minimum.
    # This probability is related to how uncertain we are about whether
    # the point beats f_best.
    
    # Uncertainty ratio: how much of the potential improvement is uncertain
    # When sigma is large relative to (f_best - mu), there's more chance
    # this point could be the new best
    uncertainty_ratio = sigma_safe / (np.abs(f_best - mu) + sigma_safe + 1e-9)
    
    # The exploration bonus should be:
    # 1. Higher when uncertainty is high (large sigma)
    # 2. Higher when we're not sure if the point beats f_best (z near 0)
    # 3. Scaled by the actual uncertainty
    
    # A refined KG approximation:
    # The term phi(z) is maximized at z=0, which corresponds to the region
    # where we're most uncertain about whether the point beats f_best.
    # This is exactly where the KG should focus exploration.
    
    # Combine EI with a KG-inspired exploration term
    # The exploration term is weighted by the PDF at z, which peaks when
    # we're most uncertain about improvement
    kg_exploration = 0.15 * phi * sigma_safe
    
    # Also add a small term that rewards high uncertainty in regions
    # where the mean is already competitive
    competitive_region = np.where(mu < f_best, 1.0, 0.0)
    bonus_for_competitive = 0.05 * uncertainty_ratio * sigma_safe * competitive_region
    
    scores = ei + kg_exploration + bonus_for_competitive
    
    # Ensure all scores are finite and non-negative
    scores = np.where(np.isfinite(scores) & (scores >= 0), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
