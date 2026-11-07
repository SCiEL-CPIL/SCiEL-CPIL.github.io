"""Acquisition function for Bayesian optimization of Goldstein-Price 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = None

def _get_elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME


def acquisition(mu, sigma, f_best):
    """
    Hybrid acquisition: combines Expected Improvement (EI) and Upper Confidence Bound (UCB)
    to balance exploitation and exploration on the Goldstein-Price 2D landscape.
    
    mu: array of shape (2000,) - posterior mean
    sigma: array of shape (2000,) - posterior std
    f_best: float - best observed value so far
    
    Returns: array of shape (2000,) - scores, higher means evaluate next
    """
    # Safety margin: return quickly if we're close to the deadline
    elapsed = _get_elapsed()
    if elapsed > 325:  # 5 seconds before the 330s limit
        # Return a simple score that favors low mu
        return -mu + 0.1 * sigma

    # Clip sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Expected Improvement component
    z_ei = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z_ei) + sigma_safe * norm.pdf(z_ei)
    # Clip EI to non-negative
    ei = np.maximum(ei, 0.0)
    
    # UCB component: explore where uncertainty is high
    # Use a beta that balances exploration
    beta = 0.2
    ucb = mu + beta * sigma_safe
    
    # For minimization, we want to minimize ucb, so score = -ucb
    # But we also want EI to drive exploitation
    # Combine: score = w_ei * ei_normalized + w_ucb * (-ucb_normalized)
    
    # Normalize EI to [0, 1] roughly
    ei_max = np.max(ei) if np.max(ei) > 0 else 1.0
    ei_norm = ei / ei_max
    
    # Normalize UCB: lower is better, so we want high score for low ucb
    ucb_min = np.min(ucb)
    ucb_max = np.max(ucb)
    ucb_range = ucb_max - ucb_min if ucb_max > ucb_min else 1.0
    # Higher score for lower ucb
    ucb_norm = 1.0 - (ucb - ucb_min) / ucb_range
    
    # Weighted combination: emphasize EI for exploitation, UCB for exploration
    # Goldstein-Price has multiple local minima, so exploration is important
    w_ei = 0.7
    w_ucb = 0.3
    
    scores = w_ei * ei_norm + w_ucb * ucb_norm
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
