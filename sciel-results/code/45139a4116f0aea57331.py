"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a Thompson Sampling-inspired
    approach combined with expected improvement per unit cost.
    
    We use a hybrid: primarily EI/cost, but with an exploration bonus
    that accounts for uncertainty, scaled by cost. This treats the
    problem as optimizing expected improvement per unit budget.
    
    mu, sigma, f_best, cost: arrays of shape (2000,)
    Returns: 2000 finite scores (higher = evaluate next)
    """
    # Deadline guard: track time to avoid exceeding 300s
    t_start = time.time()
    deadline = 300.0 - 5.0  # 5s safety margin
    
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    f_best = float(f_best)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma to avoid division by zero
    sigma = np.maximum(sigma, 1e-9)
    
    # Cost-aware Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z), where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # EI per unit cost
    ei_per_cost = ei / np.maximum(cost, 1e-9)
    
    # Add an exploration bonus: prefer points with high uncertainty
    # but penalized by cost. This helps escape local optima.
    # Use a log-scaled uncertainty bonus to keep it bounded.
    exploration_bonus = np.log1p(sigma / np.maximum(cost, 1e-9)) * 0.1
    
    # Combine: primary signal is EI/cost, secondary is exploration
    scores = ei_per_cost + exploration_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # If we're running out of time, return immediately
    if time.time() - t_start > deadline:
        return scores
    
    # Optional: if we have spare time, we could refine with a second pass,
    # but the above computation is already O(n) and fast for n=2000.
    # No additional work is needed for 2000 points.
    
    return scores
# EVOLVE-BLOCK-END
