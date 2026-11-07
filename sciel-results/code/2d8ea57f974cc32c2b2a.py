"""Rewritten program using Probability of Improvement (PoI) with cost normalization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global deadline guard
_START_TIME = time.time()
# 330s kill limit. 310s provides a 20s safety margin.
_DEADLINE = _START_TIME + 310

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Probability of Improvement (PoI).
    
    PoI(x) = P(f(x) < f_best) = Phi((f_best - mu(x)) / sigma(x))
    Score(x) = PoI(x) / cost(x)
    
    This is structurally different from LBC (linear combination) and EI (improvement magnitude).
    PoI is bounded [0,1] and focuses on the likelihood of finding a new best.
    """
    
    # Check deadline
    if time.time() > _DEADLINE:
        # Fallback: return exploration-heavy score (high sigma preferred) to finish quickly
        mu = np.asarray(mu, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        cost = np.asarray(cost, dtype=np.float64)
        sigma_safe = np.maximum(sigma, 1e-8)
        # Fallback to simple UCB-like score if time is up, but scaled down
        score = sigma_safe / (cost + 1e-9)
        return np.where(np.isfinite(score), score, 0.0)

    # Ensure inputs are numpy arrays of float64
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Handle numerical stability for sigma
    # If sigma is very small, the CDF becomes a step function.
    # We clip sigma to a small positive value to avoid division by zero.
    sigma_safe = np.maximum(sigma, 1e-8)

    # Compute z-score for Probability of Improvement
    # z = (f_best - mu) / sigma
    # If mu < f_best, z > 0, Phi(z) > 0.5 (likely improvement)
    # If mu > f_best, z < 0, Phi(z) < 0.5 (unlikely improvement)
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid extreme values causing numerical issues in CDF (though scipy is robust, 
    # clipping to [-50, 50] is safe as Phi(50) ~ 1 and Phi(-50) ~ 0)
    z = np.clip(z, -50, 50)
    
    # Compute Probability of Improvement
    # This gives a score in [0, 1]
    poi = norm.cdf(z)
    
    # Cost-aware normalization: divide by cost to prefer cheaper points
    # Cost is typically in [1, 2] for this task (1 + mean(x_scaled))
    score = poi / (cost + 1e-9)
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
