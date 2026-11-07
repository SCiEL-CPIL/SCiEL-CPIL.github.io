"""Cost-aware acquisition function for Bayesian optimization (griewank 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START = time.time()
_LIMIT = 325.0  # Safety margin before the 330 s wall-clock kill

def acquisition(mu, sigma, f_best, cost):
    """
    Deterministic Expected Improvement (EI) with cost normalization.
    
    Strategy:
    1. Compute Expected Improvement: EI(x) = E[max(f_best - f(x), 0)]
       where f(x) ~ N(mu, sigma^2).
    2. Normalize by cost to get EI/cost.
    3. Return scores (higher = better).
    
    This is a deterministic, robust baseline that balances exploitation (high EI)
    and exploration (high sigma) via the EI formula, while penalizing expensive
    evaluations via the cost normalization.
    
    EI(x) = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure shapes and safety
    cost = np.maximum(cost, 1e-6)
    sigma = np.maximum(sigma, 1e-9)
    
    # Deadline guard: if we are close to the limit, fall back to a fast deterministic score
    if time.time() - _START > _LIMIT:
        # Deterministic fallback: -mu / cost (simple exploitation)
        scores = -mu / cost
        return np.where(np.isfinite(scores), scores, 0.0)
    
    # Compute Expected Improvement
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is the standard normal CDF and phi is the standard normal PDF
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure EI is non-negative (should be, but numerical issues can occur)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score: EI / cost
    scores = ei / cost
    
    # Clip scores to prevent extreme values
    scores = np.clip(scores, 0.0, 100.0)
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
