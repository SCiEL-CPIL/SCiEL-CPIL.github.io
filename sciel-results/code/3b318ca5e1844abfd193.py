"""Acquisition function for Bayesian optimization of Styblinski-Tang 1D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Cache for elapsed time tracking
_start_time = [None]

def acquisition(mu, sigma, f_best):
    """
    Combined acquisition: normalized EI + inverted UCB component.
    
    mu: array of shape (2000,) - GP mean predictions
    sigma: array of shape (2000,) - GP standard deviations
    f_best: float - best observed value so far
    
    Returns: array of shape (2000,) - scores (higher = more promising)
    """
    if _start_time[0] is None:
        _start_time[0] = time.time()
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Expected Improvement component
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Normalize EI to [0, 1] range for balanced combination
    ei_max = np.max(ei)
    if ei_max > 1e-12:
        ei_norm = ei / ei_max
    else:
        ei_norm = np.zeros_like(ei)
    
    # UCB component: exploration weighted by uncertainty
    # Use a time-adaptive beta that starts high and decreases
    elapsed = time.time() - _start_time[0]
    # Beta decays from ~3.0 to ~1.0 over the budget
    beta = 3.0 * np.exp(-0.5 * elapsed / 60.0) + 1.0
    
    ucb = mu + beta * sigma_safe
    
    # Invert UCB since we're minimizing (lower mu is better, higher sigma is exploratory)
    # For minimization: we want low mu and high sigma
    # So the "score" from UCB perspective is: -mu + beta * sigma
    ucb_score = -mu + beta * sigma_safe
    
    # Normalize UCB score
    ucb_min = np.min(ucb_score)
    ucb_max = np.max(ucb_score)
    if ucb_max - ucb_min > 1e-12:
        ucb_norm = (ucb_score - ucb_min) / (ucb_max - ucb_min)
    else:
        ucb_norm = np.zeros_like(ucb_score)
    
    # Combine: weighted sum of normalized EI and normalized UCB score
    # Early on: more exploration (UCB weight higher)
    # Later: more exploitation (EI weight higher)
    progress = min(elapsed / 300.0, 1.0)
    w_ei = 0.5 + 0.3 * progress       # 0.5 to 0.8
    w_ucb = 1.0 - w_ei                # 0.5 to 0.2
    
    scores = w_ei * ei_norm + w_ucb * ucb_norm
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
