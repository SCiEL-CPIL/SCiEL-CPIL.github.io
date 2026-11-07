# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard: 330s wall-clock limit, use 325s as safety margin
_DEADLINE = 325.0
_START_TIME = time.time()

def _check_deadline():
    """Return True if we should abort and return current best."""
    return (time.time() - _START_TIME) > _DEADLINE

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Thompson Sampling.
    
    Instead of a deterministic acquisition like EI or UCB, we sample a function
    from the GP posterior and score candidates based on the sampled value divided
    by cost. This naturally balances exploration and exploitation and is robust
    to multi-modal landscapes.
    """
    # Check deadline first
    if _check_deadline():
        return np.zeros_like(mu, dtype=np.float64)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    
    # Clip to reasonable ranges
    sigma = np.clip(sigma, 1e-8, 1e4)
    cost = np.clip(cost, 1.0, 2.0)
    
    # Thompson Sampling: Sample a function value for each candidate
    # f_s(x) ~ N(mu(x), sigma(x)^2)
    # We use a single set of standard normal samples z
    z = np.random.standard_normal(len(mu))
    
    # Sampled function values
    f_sampled = mu + sigma * z
    
    # Score: We want to minimize f_sampled / cost
    # So the acquisition score (higher is better) is -f_sampled / cost
    score = -f_sampled / cost
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
