"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global start time for deadline guard
_start_time = time.time()
_DEADLINE = _start_time + 290.0  # 10s safety margin before 300s limit

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware UCB acquisition function for minimization.
    
    Args:
        mu: GP mean predictions (2000,)
        sigma: GP standard deviation predictions (2000,)
        f_best: Best observed value so far
        cost: Cost of evaluating each candidate (2000,)
        
    Returns:
        scores: Acquisition scores (2000,), higher is better
    """
    # Time guard: if we are close to the limit, return a simple fallback
    if time.time() > _DEADLINE:
        sigma_safe = np.maximum(sigma, 1e-8)
        cost_safe = np.maximum(cost, 1e-8)
        # Fallback UCB
        scores = (2.0 * sigma_safe - mu) / np.log(1.0 + cost_safe)
        scores[~np.isfinite(scores)] = 0.0
        return scores

    # Convert to numpy arrays for vectorized operations
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Numerical safety
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)
    
    # UCB for minimization:
    # We want to minimize f(x).
    # Lower Confidence Bound (LCB) = mu - beta * sigma
    # We maximize the negative LCB: Score = -(mu - beta * sigma) = beta * sigma - mu
    # Using beta = 2.0 to encourage exploration given the small budget.
    beta = 2.0
    
    # Cost-aware score:
    # Using log(1 + cost) to soften the penalty of higher cost points.
    # This prevents the algorithm from completely ignoring high-cost regions
    # which might contain the global minimum, while still preferring cheaper points.
    log_cost = np.log(1.0 + cost_safe)
    
    scores = (beta * sigma_safe - mu) / log_cost
    
    # Ensure all outputs are finite
    scores[~np.isfinite(scores)] = 0.0
    
    return scores
# EVOLVE-BLOCK-END
