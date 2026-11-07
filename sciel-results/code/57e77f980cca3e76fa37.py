"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization of Hartmann 3D.
    
    Structural change: Use a cost-adjusted UCB that properly balances exploration
    and exploitation while accounting for the cost structure. The key insight is
    that we should NOT penalize cost aggressively since the minimum is uniformly
    distributed in the inner 80% of the domain. Instead, we use a mild cost
    adjustment and focus on the UCB for exploration/exploitation balance.
    """
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 330s kill
    
    # Ensure inputs are valid
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    f_best = float(f_best)
    
    n = len(mu)
    
    # Basic safety: if anything is wrong, return a simple UCB
    if n == 0 or not np.all(np.isfinite(mu)) or not np.all(np.isfinite(sigma)):
        return np.zeros(n, dtype=np.float64)
    
    # Clamp sigma to positive values
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Compute UCB (Upper Confidence Bound) for minimization
    # UCB = mu - beta * sigma
    # We want to maximize this, so higher UCB means better candidate
    
    # Adaptive beta: start with higher exploration, decrease over time
    # Estimate progress based on how good f_best is relative to typical Hartmann 3D values
    # Hartmann 3D minimum is about -3.86, so if f_best is close, we should exploit more
    
    # Use a fixed beta that balances exploration and exploitation well
    # The cost structure is mild (cost in [1, 2]), so we don't need strong cost penalty
    beta = 1.5  # Moderate exploration
    
    # UCB score (higher is better for minimization)
    ucb = mu - beta * sigma_safe
    
    # Cost adjustment: mild penalty for expensive points
    # Cost is in [1, 2], so (cost - 1) is in [0, 1]
    # Use a small weight to not dominate the UCB
    cost_penalty_weight = 0.1
    cost_adjusted = ucb - cost_penalty_weight * (cost - 1.0)
    
    # Ensure finiteness
    cost_adjusted = np.where(np.isfinite(cost_adjusted), cost_adjusted, -1e10)
    
    # Check time
    if time.time() > deadline - 2.0:
        # Fallback to simple UCB if running out of time
        cost_adjusted = mu - beta * sigma_safe
        cost_adjusted = np.where(np.isfinite(cost_adjusted), cost_adjusted, -1e10)
    
    return cost_adjusted
# EVOLVE-BLOCK-END
