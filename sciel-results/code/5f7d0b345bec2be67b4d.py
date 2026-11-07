"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard: 330s wall-clock limit, use 325s as safety margin
_DEADLINE = time.time() + 325.0


def _time_left():
    return _DEADLINE - time.time()


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses a Cost-Weighted Thompson Sampling proxy:
    - Sample a global multiplier lambda from a Beta distribution
    - Score candidates as (lambda * (f_best - mu) + (1-lambda) * sigma) / cost
    
    This stochastically shifts the exploration-exploitation trade-off per iteration.
    """
    # Check deadline - if time is running out, return a safe fallback
    if _time_left() < 5.0:
        # Fallback: simple cost-weighted improvement
        safe_sigma = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / safe_sigma
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return ei / cost

    # Handle edge cases
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Sample lambda from Beta distribution for Thompson Sampling proxy
    # Beta(1, 1) is uniform; we use a slightly biased distribution
    # to balance exploration and exploitation
    # As the algorithm progresses (higher cost), we can shift toward exploitation
    mean_cost = np.mean(cost)
    # Use a simple heuristic: earlier iterations favor exploration more
    # Beta(alpha, beta) where alpha controls exploitation, beta controls exploration
    alpha = 2.0
    beta = 1.5
    lam = np.random.beta(alpha, beta)
    
    # Compute acquisition score
    # Exploitation component: expected improvement (f_best - mu)
    # Exploration component: uncertainty (sigma)
    exploit = f_best - mu
    explore = sigma_safe
    
    # Combine with Thompson Sampling weight
    score = lam * exploit + (1.0 - lam) * explore
    
    # Normalize by cost (higher cost = lower priority)
    score = score / np.maximum(cost, 1e-9)
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
