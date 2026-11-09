import numpy as np
from scipy.stats import norm
import time

# Track state for adaptive exploration
_start_time = None
_eval_count = 0

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Hartmann 6D.
    Uses GP-LCB with adaptive exploration coefficient that increases
    with the number of evaluated points to prevent premature convergence.
    """
    global _start_time, _eval_count
    
    # Initialize timing if needed
    if _start_time is None:
        _start_time = time.time()
        _eval_count = 0
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Handle zero sigma to avoid division by zero
    safe_sigma = np.where(sigma < 1e-12, 1e-12, sigma)
    
    # Adaptive exploration coefficient: beta = 1 + 0.1 * t
    # where t is the number of evaluations so far
    # This increases exploration as we gather more information
    t = _eval_count
    beta = 1.0 + 0.1 * t
    
    # GP-LCB: score = -mu + beta * sigma
    # We want higher scores to be better, so we negate
    # But since we're minimizing, lower mu is better
    # LCB = mu - beta * sigma (lower is more promising)
    # For maximization: score = -mu + beta * sigma
    scores = -mu + beta * safe_sigma
    
    # Increment evaluation count
    _eval_count += 1
    
    # Ensure all scores are finite to satisfy the contract
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
