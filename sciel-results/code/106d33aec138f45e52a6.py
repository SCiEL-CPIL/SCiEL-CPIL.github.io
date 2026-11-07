"""Acquisition function for Bayesian optimization of branin 2d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Probability of Improvement (PoI) acquisition for minimization.
    
    PoI(x) = P(f(x) < f_best) = Phi((f_best - mu(x)) / sigma(x))
    
    This favors broad coverage of the multi-modal Branin landscape by
    decoupling exploration from the magnitude of potential improvement.
    
    Parameters:
        mu: (2000,) array of GP means in objective units.
        sigma: (2000,) array of GP standard deviations in objective units.
        f_best: Lowest observed function value.
    
    Returns:
        (2000,) array of scores, higher meaning more desirable.
    """
    # Compute z-score for probability of improvement
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # For minimization: P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma_safe
    
    # PoI is the CDF of the standard normal at z
    poi = norm.cdf(z)
    
    # Add a small exploration bonus for high uncertainty to encourage
    # visiting unexplored regions, but keep it modest so that PoI
    # remains the dominant signal
    # The bonus is scaled by sigma relative to the typical scale of f_best
    # Branin values range from ~0.4 to ~300, so use a scale-aware bonus
    scale = max(f_best, 1.0)
    exploration_bonus = 0.05 * (sigma_safe / scale)
    
    scores = poi + exploration_bonus
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
