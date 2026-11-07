"""Bayesian optimization acquisition for Beale 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_obs_count = 5
_start_time = None

def acquisition(mu, sigma, f_best):
    """
    Combined GP-UCB (time-varying beta) and normalized Expected Improvement acquisition function.
    
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
    global _obs_count, _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: if we're close to the limit, return a safe score
    # (This is a safety net; the actual computation is fast)
    elapsed = time.time() - _start_time
    if elapsed > 325:  # 5 seconds before the 330s limit
        return np.zeros_like(mu)
    
    t = _obs_count
    _obs_count += 1
    
    # Ensure inputs are finite
    mu = np.where(np.isfinite(mu), mu, f_best + 1e6)
    sigma = np.maximum(sigma, 1e-9)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    
    # Time-varying beta for GP-UCB: beta_t = sqrt(2*t*log(1+t))
    # This provides theoretically grounded exploration that scales with evaluations
    beta = np.sqrt(2.0 * t * np.log(1.0 + t))
    
    # GP-UCB score for minimization: we want to maximize (-mu + sqrt(beta) * sigma)
    # The UCB bound on the minimum is mu - sqrt(beta)*sigma, so we maximize its negative
    ucb_score = -mu + np.sqrt(beta) * sigma
    
    # Expected Improvement for minimization
    z = (f_best - mu) / sigma
    z_clipped = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma * norm.pdf(z_clipped)
    ei = np.maximum(ei, 0.0)  # EI should be non-negative
    
    # Normalize both components to [0, 1] for balanced combination
    ucb_min = np.min(ucb_score)
    ucb_max = np.max(ucb_score)
    ucb_range = ucb_max - ucb_min
    if ucb_range > 1e-12:
        ucb_norm = (ucb_score - ucb_min) / ucb_range
    else:
        ucb_norm = np.zeros_like(ucb_score)
    
    ei_max = np.max(ei)
    if ei_max > 1e-12:
        ei_norm = ei / ei_max
    else:
        ei_norm = np.zeros_like(ei)
    
    # Combine: 70% UCB + 30% EI
    # The 70/30 split favors exploration early on (when beta is small)
    # but as beta grows with t, exploration naturally increases
    scores = 0.7 * ucb_norm + 0.3 * ei_norm
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
