"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using rank-based Pareto front selection.
    
    Computes Expected Improvement (EI) and a cost-aware objective, then
    assigns scores based on non-dominated ranking to enforce a strict
    trade-off between improvement potential and evaluation cost.
    
    Parameters:
    -----------
    mu : array_like, shape (2000,)
        GP mean predictions at candidate points.
    sigma : array_like, shape (2000,)
        GP standard deviation predictions at candidate points.
    f_best : float
        Best (lowest) observed function value so far.
    cost : array_like, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = more desirable to evaluate).
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: clip sigma to avoid division by zero
    sigma = np.clip(sigma, 1e-9, None)
    
    # Clip cost to avoid division by zero
    cost = np.clip(cost, 1e-9, None)
    
    # Compute Expected Improvement
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Add a small UCB exploration component to help escape local optima
    beta = 0.5
    ucb = ei + beta * sigma
    
    # Objective 1: maximize expected improvement (higher is better)
    # Objective 2: minimize cost (lower is better)
    
    # For Pareto dominance, we want to maximize both objectives.
    # So we negate cost to turn it into a maximization problem.
    obj1 = ucb  # maximize
    obj2 = -cost  # maximize (i.e., minimize cost)
    
    # Compute non-dominated ranks using a simple O(n^2) approach for 2000 points
    n = len(mu)
    ranks = np.zeros(n, dtype=np.int32)
    dominated_count = np.zeros(n, dtype=np.int32)
    dominating_count = np.zeros(n, dtype=np.int32)
    
    # For each pair, determine dominance
    # A dominates B if A is at least as good in all objectives and strictly better in at least one
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # Check if i dominates j
            if (obj1[i] >= obj1[j] and obj2[i] >= obj2[j]) and \
               (obj1[i] > obj1[j] or obj2[i] > obj2[j]):
                dominated_count[i] += 1
                dominating_count[j] += 1
    
    # Assign ranks: non-dominated points get rank 0, etc.
    # Simple approach: sort by dominated_count, then by dominated_count + dominating_count
    # Better: iterative rank assignment
    
    # Iterative rank assignment
    remaining = np.ones(n, dtype=bool)
    current_rank = 0
    while np.any(remaining):
        # Find non-dominated points among remaining
        front_mask = np.zeros(n, dtype=bool)
        for i in np.where(remaining)[0]:
            is_non_dominated = True
            for j in np.where(remaining)[0]:
                if i == j:
                    continue
                if (obj1[j] >= obj1[i] and obj2[j] >= obj2[i]) and \
                   (obj1[j] > obj1[i] or obj2[j] > obj2[i]):
                    is_non_dominated = False
                    break
            front_mask[i] = is_non_dominated
        
        if not np.any(front_mask):
            # Fallback: assign all remaining to current rank
            front_mask = remaining
        
        ranks[front_mask] = current_rank
        remaining[front_mask] = False
        current_rank += 1
    
    # Convert ranks to scores: lower rank (better) gets higher score
    # Use exponential decay to create a smooth scoring function
    max_rank = max(ranks.max(), 1)
    decay = 0.3
    scores = np.exp(-decay * ranks)
    
    # Add a small tiebreaker based on actual EI value to differentiate within same rank
    ei_normalized = ei / (ei.max() + 1e-12)
    scores = scores * (0.9 + 0.1 * ei_normalized)
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
