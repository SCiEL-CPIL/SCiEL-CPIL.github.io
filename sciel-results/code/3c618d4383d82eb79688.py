import numpy as np
from scipy.stats import norm
import time

_start_time = None
_DEADLINE = 320.0  # Safety margin before 330s kill

def acquisition(mu, sigma, f_best, cost):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        # Fallback: Simple, fast score
        # Use a very simple heuristic to avoid hanging
        # Prefer low mu and low cost
        cost_safe = np.maximum(cost, 1e-9)
        return -mu / cost_safe

    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Calculate z-score
    # z = (f_best - mu) / sigma
    # If mu < f_best, z > 0 (improvement likely)
    # If mu > f_best, z < 0 (improvement unlikely)
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Clip z to avoid overflow in pdf/cdf? Phi and phi are stable for large z.
    # phi(z) underflows for z > ~40, which is fine (becomes 0).
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Exploration Bonus
    # Add a cost-scaled exploration term
    # The cell finding suggests "cost-scaled exploration bonus"
    # This is effectively adding beta * sigma to the numerator, then dividing by cost
    beta = 2.0
    exploration_bonus = beta * sigma_safe
    
    # Combined Score
    # Higher score = better
    # We want high EI and high Exploration, low Cost
    score = (ei + exploration_bonus) / cost_safe
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
