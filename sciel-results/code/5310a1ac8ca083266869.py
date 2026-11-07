"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 330.0  # wall-clock seconds total allowed


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Styblinski-Tang 1D.
    
    Uses Expected Improvement (EI) with a well-tuned balance for the 
    multi-modal Styblinski-Tang function.
    
    Args:
        mu: array of shape (2000,) with posterior means
        sigma: array of shape (2000,) with posterior standard deviations
        f_best: lowest observed function value
    
    Returns:
        array of shape (2000,) with acquisition scores (higher is better)
    """
    # Check deadline
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE - 5.0:
        # Return a simple fallback score
        return -mu + 0.1 * sigma
    
    # Guard against negative or zero sigma
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Compute z-score: (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Standard Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # For Styblinski-Tang, pure EI works well but can benefit from slight
    # exploration boost. Let's use a modified approach:
    # 1. Primary: EI (exploit)
    # 2. Secondary: small UCB component to escape local optima
    
    # Normalize EI to [0, 1]
    ei_min = np.min(ei)
    ei_max = np.max(ei)
    ei_range = ei_max - ei_min + 1e-9
    ei_normalized = (ei - ei_min) / ei_range
    
    # UCB component: balance between low mu (good) and high sigma (uncertain)
    # Use a beta parameter that balances exploitation and exploration
    beta = 0.5  # exploration parameter
    
    # UCB score: -mu + beta * sigma (higher is better)
    ucb = -mu + beta * sigma_safe
    
    # Normalize UCB to [0, 1]
    ucb_min = np.min(ucb)
    ucb_max = np.max(ucb)
    ucb_range = ucb_max - ucb_min + 1e-9
    ucb_normalized = (ucb - ucb_min) / ucb_range
    
    # Combine: weight EI heavily with a small UCB boost
    # For Styblinski-Tang's multi-modal nature, we want strong exploitation
    # but enough exploration to find the global minimum
    ei_weight = 0.85
    ucb_weight = 0.15
    
    score = ei_weight * ei_normalized + ucb_weight * ucb_normalized
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
