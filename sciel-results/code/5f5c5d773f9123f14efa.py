"""Cost-aware acquisition for Powell 4D BO with non-uniform evaluation costs."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Module-level state for time tracking
_start_time = None

def _get_elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Powell 4D BO.
    
    Uses a multiplicative Probability of Improvement (PI) with cost exponentiation,
    combined with a small UCB exploration term scaled by the current GP uncertainty.
    
    Parameters:
    - mu: (2000,) array of GP mean predictions
    - sigma: (2000,) array of GP std predictions  
    - f_best: scalar, best observed value so far
    - cost: (2000,) array of evaluation costs for each candidate
    
    Returns:
    - (2000,) array of acquisition scores (higher = better)
    """
    # Time guard: if we've been running too long, return safe fallback
    elapsed = _get_elapsed()
    if elapsed > 280:  # Safety margin before 300s limit
        mu = np.asarray(mu)
        sigma = np.asarray(sigma)
        cost = np.asarray(cost)
        z = (f_best - mu) / (sigma + 1e-9)
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return ei / (cost + 1e-9)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safe sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Z-score for Probability of Improvement
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement: P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    pi = norm.cdf(z)
    
    # Cost-aware weighting via exponentiation:
    # PI^(1/cost) naturally penalizes expensive evaluations.
    # When cost is low (cheap), exponent is high, requiring high PI.
    # When cost is high (expensive), exponent is low, allowing moderate PI.
    cost_safe = np.maximum(cost, 1e-6)
    pi_cost_aware = np.power(np.maximum(pi, 1e-15), 1.0 / cost_safe)
    
    # Exploration term: UCB-like bonus scaled by GP uncertainty
    # This helps escape local minima and cover the space more effectively.
    # We scale the exploration by the mean sigma to keep it adaptive.
    mean_sigma = np.mean(sigma)
    exploration_lambda = 0.3 + 0.3 * min(1.0, mean_sigma / (np.std(mu) + 1e-9))
    exploration_term = exploration_lambda * sigma / cost_safe
    
    # Combine exploitation (PI-based) and exploration (sigma-based)
    # The PI term is bounded in [0, 1], so we scale it to be the dominant term
    score = pi_cost_aware + exploration_term
    
    # Ensure finite values
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
# EVOLVE-BLOCK-END
