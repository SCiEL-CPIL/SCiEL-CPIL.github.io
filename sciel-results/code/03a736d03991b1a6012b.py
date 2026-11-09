"""Acquisition function for Bayesian optimization of beale 2d."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_DEADLINE = 300.0  # Leave margin before 330s kill


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Upper Confidence Bound (UCB).
    
    Structural Change:
    Replaces the hybrid EI/UCB mixture with a pure UCB strategy.
    UCB is linear in mu and sigma, providing a different exploration/exploitation
    trade-off compared to the nonlinear EI. This is particularly effective for the
    Beale function's wide flat regions where EI might underestimate the potential
    for large improvements or get stuck in local optima due to its non-linear scaling.
    
    We use an adaptive beta that is higher when f_best is large (early in optimization)
    and lower when f_best is small (late in optimization).
    """
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        return -mu
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Adaptive beta: 
    # Base exploration of 1.5, plus a term that adds up to 0.5 more when f_best is large.
    # f_best / (f_best + 1) is ~1 for large f_best, ~0 for small f_best.
    f_best_clamped = max(f_best, 1e-12)
    beta = 1.5 + 0.5 * (f_best_clamped / (f_best_clamped + 1.0))
    
    # UCB for minimization: -mu + beta * sigma
    score = -mu + beta * sigma_safe
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
