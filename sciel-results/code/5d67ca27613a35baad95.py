"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 325.0  # safety margin before 330s kill


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of a 2D RBF mixture.
    
    Uses Upper Confidence Bound (UCB) for minimization (Lower Confidence Bound logic).
    We want to minimize f. The "score" returned should be high for points likely to
    have low f values or high uncertainty.
    
    Score = beta * sigma - mu
    
    Parameters:
        mu: array of shape (2000,), posterior mean at candidates
        sigma: array of shape (2000,), posterior std at candidates
        f_best: scalar, best (lowest) observed value so far
    
    Returns:
        array of shape (2000,), scores (higher = more promising to evaluate next)
    """
    # Deadline guard: if we're close to the time limit, return a simple score quickly
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        # Fallback: return something finite and simple (prefer low mu)
        return -mu
    
    # Ensure sigma is positive and finite
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # UCB for minimization:
    # We want to select x that minimizes mu(x) - beta * sigma(x).
    # Since the evaluator picks the MAX score, we return:
    # Score = -(mu(x) - beta * sigma(x)) = beta * sigma(x) - mu(x)
    
    # Choose beta. A value around 1.0 to 3.0 is typical.
    # Let's try 2.0 to encourage exploration.
    beta = 2.0
    
    scores = beta * sigma_safe - mu
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
