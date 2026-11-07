"""Bayesian optimization acquisition for Beale 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

_obs_count = 5

def acquisition(mu, sigma, f_best):
    """
    Combined GP-UCB (LCB for minimization) and Expected Improvement acquisition function.
    
    Parameters
    ----------
    mu : ndarray, shape (2000,)
        GP posterior mean at candidate points (in objective units).
    sigma : ndarray, shape (2000,)
        GP posterior standard deviation at candidate points (in objective units).
    f_best : float
        Best observed value so far.
    
    Returns
    -------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher values are preferred.
    """
    global _obs_count
    t = _obs_count
    _obs_count += 1

    # Exploration parameter beta
    # Using a moderate constant beta to balance exploration/exploitation
    # Theoretical GP-UCB beta grows, but for small N, a constant is often more stable
    beta = 2.0

    # LCB score for minimization: we want to maximize (-mu + beta * sigma)
    # This is equivalent to minimizing (mu - beta * sigma)
    ucb_score = -mu + beta * sigma

    # Standardized EI for minimization
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)

    # Combine UCB and EI
    # Both ucb_score and ei are higher for better (lower f) points.
    # We add them. EI provides exploitation pressure near f_best.
    # UCB provides exploration.
    scores = ucb_score + ei

    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END
