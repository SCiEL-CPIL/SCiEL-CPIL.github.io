"""Acquisition function for Bayesian optimization of Branin 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global state for tracking call count and time
_call_count = 0
_start_time = None


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Branin 2D.
    
    Uses Probability of Improvement (PI) with a small uncertainty floor to
    prevent premature convergence in the multimodal Branin landscape.
    PI directly maximizes the probability of finding a new minimum, which is
    more targeted than UCB's uniform exploration.
    
    Args:
        mu: array of shape (n,) - predicted means
        sigma: array of shape (n,) - predicted standard deviations
        f_best: float - best observed value so far
    
    Returns:
        scores: array of shape (n,) - acquisition scores (higher = better)
    """
    global _call_count, _start_time
    
    if _start_time is None:
        _start_time = time.time()
    _call_count += 1
    
    # Deadline guard: if we're close to the 330s limit, return a safe fallback
    elapsed = time.time() - _start_time
    if elapsed > 325:
        # Return a simple LCB to finish quickly
        sigma_safe = np.maximum(sigma, 1e-9)
        return -mu + 1.0 * sigma_safe
    
    # Ensure sigma is positive and finite
    sigma_safe = np.maximum(sigma, 1e-9)
    sigma_safe = np.where(np.isfinite(sigma_safe), sigma_safe, 1e-9)
    
    # Total evaluations: 30 (5 initial + 25 BO steps)
    # So we have 25 acquisition calls (after the first 5 points)
    if _call_count <= 25:
        progress = _call_count / 25.0
    else:
        progress = 1.0
    
    # Probability of Improvement: P(X < f_best) where X ~ N(mu, sigma^2)
    # z = (f_best - mu) / sigma
    # PI = Phi(z)
    
    z = (f_best - mu) / sigma_safe
    
    # PI: probability that the candidate improves on f_best
    pi = norm.cdf(z)
    
    # Add a small uncertainty floor to prevent premature convergence
    # In high-uncertainty regions, PI can be close to 0.5 even when mu is
    # far from f_best. A small sigma boost encourages exploring uncertain areas
    # where a better minimum might exist.
    # The boost decays as we progress (less exploration needed later)
    uncertainty_weight = 0.05 * (1.0 - progress)
    uncertainty_boost = uncertainty_weight * sigma_safe
    
    scores = pi + uncertainty_boost
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
