"""Cost-aware acquisition function for Bayesian optimization (ackley 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def _probability_of_improvement(mu, sigma, f_best):
    """
    Probability of Improvement: P(f(x) < f_best)
    where f(x) ~ N(mu, sigma^2).
    
    This is a fundamentally different acquisition criterion from EI:
    - PI is bounded in [0, 1]
    - It measures where the CDF sits relative to f_best
    - It naturally balances exploration (high sigma) and exploitation (low mu)
    - Unlike EI, it doesn't reward large expected improvements from far-away points
    """
    s = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / s
    pi = norm.cdf(z)
    return np.where(np.isfinite(pi), np.clip(pi, 0.0, 1.0), 0.0)


def acquisition(mu, sigma, f_best, cost):
    # Deadline guard
    t0 = time.time()
    
    s = np.maximum(sigma, 1e-9)
    c = np.maximum(cost, 1e-9)
    
    # Compute Probability of Improvement
    pi = _probability_of_improvement(mu, sigma, f_best)
    
    # Structural change: use PI instead of EI
    # PI has different properties:
    # 1. Bounded [0,1] - no need for thinning penalty
    # 2. Naturally penalizes points far from f_best even with high sigma
    # 3. Creates a smoother landscape for selection
    
    # Cost-aware weighting: use a power that balances exploration vs cost
    # PI is already in [0,1], so we can use a different cost penalty structure
    # Using cost^0.6 to be slightly more permissive than EI/cost^0.8
    score = pi / (c ** 0.6)
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Safety margin: if we've spent too much time, return immediately
    elapsed = time.time() - t0
    if elapsed > 280:  # 20s buffer before 300s limit
        return score
    
    return score
# EVOLVE-BLOCK-END
