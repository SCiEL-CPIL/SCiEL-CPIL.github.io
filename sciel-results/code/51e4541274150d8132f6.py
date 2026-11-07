"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Module-level cache for time budget
_t0 = None

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization (hartmann 3d).
    
    Uses a combination of:
    1. Expected Improvement (EI) for exploitation
    2. Uncertainty bonus (UCB-like) for exploration
    3. Cost normalization
    
    The cost-aware term penalizes expensive points (higher mean(x)) more heavily.
    We also add a small exploration term that favors points with high sigma,
    especially in early iterations when f_best is not yet well-estimated.
    """
    global _t0
    if _t0 is None:
        _t0 = time.time()
    
    # Safety margin: if we're getting close to the 330s limit, return a simple score
    elapsed = time.time() - _t0
    if elapsed > 320:
        # Fallback: just use -mu / cost to avoid any slow computation
        return -mu / (cost + 1e-9)
    
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clamp sigma to avoid division by zero and numerical issues
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Expected Improvement (EI)
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    # Clip z to avoid extreme values
    z = np.clip(z, -10, 10)
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost-normalized EI
    # cost is 1 + mean(x), so it's in [1, 2]
    # We want to penalize high-cost points
    ei_per_cost = ei / (cost + 1e-9)
    
    # Exploration term: UCB-like bonus for high uncertainty
    # The coefficient balances exploitation vs exploration
    # Use a smaller exploration weight as we get more data (proxy: if f_best is very negative, we're doing well)
    # For hartmann3d, the minimum is around -3.86
    # If f_best is close to the optimum, reduce exploration
    exploration_weight = 0.3
    
    # UCB term: mu + beta * sigma
    # But since we're minimizing, we want to explore where sigma is high
    # The exploration bonus should be proportional to sigma
    ucb_bonus = exploration_weight * sigma_safe
    
    # Combine: primarily use EI per cost, add exploration bonus
    # Scale the exploration bonus similarly by cost to be consistent
    score = ei_per_cost + ucb_bonus / (cost + 1e-9)
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Return the scores (higher is better)
    return score
# EVOLVE-BLOCK-END
