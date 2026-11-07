"""Acquisition function for Bayesian optimization of Goldstein-Price 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Track time to ensure we don't exceed the 300s limit (with safety margin)
_start_time = None


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Goldstein-Price 2D.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean of the GP posterior at candidate points.
    sigma : ndarray, shape (2000,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best (lowest) observed value so far.
        
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores, higher meaning more promising.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: if we're getting close to the 300s limit, return quickly
    elapsed = time.time() - _start_time
    if elapsed > 280:  # 20s safety margin
        # Fallback: Simple UCB
        sigma_safe = np.maximum(sigma, 1e-9)
        scores = -mu + 1.0 * sigma_safe
        return np.where(np.isfinite(scores), scores, 0.0)
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Calculate Upper Confidence Bound (UCB) score
    # For minimization, we want to minimize mu - beta * sigma
    # So the "score" (higher is better) is -mu + beta * sigma
    # We need to choose beta.
    
    # Adaptive Beta:
    # Start with high exploration, reduce as we get closer to the known minimum (3.0)
    # f_best starts around 5-10 (random init) and drops to 3.
    # progress = (start_val - f_best) / (start_val - min_val)
    # We don't know start_val exactly, but we can assume a range.
    # Let's use a fixed decay based on f_best.
    # If f_best is high, beta is high. If f_best is low, beta is low.
    
    # Heuristic: 
    # At start (f_best ~ 5), beta ~ 2.0
    # At end (f_best ~ 3), beta ~ 0.5
    # Linear interpolation
    f_min = 3.0
    f_start_est = 5.0 # Estimated initial best from 5 random points
    
    # Clip progress to [0, 1]
    if f_start_est > f_min:
        progress = np.clip((f_start_est - f_best) / (f_start_est - f_min), 0.0, 1.0)
    else:
        progress = 0.0
        
    # Beta decays from 2.0 to 0.5
    beta = 2.0 - 1.5 * progress
    
    # UCB Score (higher is better)
    ucb_score = -mu + beta * sigma_safe
    
    # Combine EI and UCB
    # EI is in objective units. UCB is in objective units.
    # They are comparable in scale.
    # Weight EI heavily for exploitation, UCB for exploration.
    # Since EI is 0 when mu > f_best, it naturally focuses on improvement.
    # UCB is always non-zero and pushes for exploration.
    
    # Let's use a weighted sum.
    # w_ei = 1.0
    # w_ucb = 0.5 (tunable)
    
    scores = ei + 0.5 * ucb_score
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
