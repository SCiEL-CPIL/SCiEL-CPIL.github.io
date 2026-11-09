"""Cost-aware acquisition function for Bayesian optimization (griewank 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_DEADLINE = 320.0  # safety margin before 330s kill
_START_TIME = None

def _ensure_start():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.monotonic()

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for minimizing griewank 2d.
    
    Uses a hybrid of Expected Improvement (EI) and a cost-aware UCB exploration
    bonus. The EI term captures exploitation, while the UCB term encourages
    exploration in cheap, uncertain regions.
    
    Parameters:
        mu: (2000,) GP mean predictions in objective units
        sigma: (2000,) GP std predictions in objective units
        f_best: float, best observed value
        cost: (2000,) cost of evaluating each candidate
    
    Returns:
        (2000,) acquisition scores (higher = evaluate next)
    """
    _ensure_start()
    
    # Deadline guard: if we're close to the limit, return a safe fallback
    if time.monotonic() - _START_TIME > _DEADLINE:
        # Fallback: simple EI/cost
        z = (f_best - mu) / (sigma + 1e-9)
        return ((f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)) / (cost + 1e-9)
    
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma to avoid numerical issues
    sigma = np.maximum(sigma, 1e-6)
    cost = np.maximum(cost, 1e-6)
    
    # Expected Improvement (EI) for minimization
    improvement = f_best - mu
    z = improvement / sigma
    ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Cost-aware UCB exploration bonus
    # In cheap regions, we can afford more exploration
    ucb_bonus = sigma / cost
    
    # Hybrid: EI/cost + exploration bonus
    # The EI/cost term handles exploitation, the bonus handles exploration
    # Scale the bonus to be comparable to EI/cost
    ei_cost = ei / cost
    bonus = 0.1 * ucb_bonus  # small weight for exploration
    
    score = ei_cost + bonus
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
