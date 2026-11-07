"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition using a Pareto-frontier based non-dominated sort.
    
    We score candidates by their rank in a non-dominated sort of 
    (predicted value, evaluation cost). Points on the cost-efficient frontier
    where marginal reduction in predicted value per unit cost is maximized
    get the highest scores.
    """
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = len(mu)
    scores = np.zeros(n, dtype=np.float64)
    
    # Handle edge cases
    if n == 0:
        return scores
    
    # Clip sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute expected improvement (EI) as a proxy for predicted value improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # EI is the expected reduction in objective value
    # We want to maximize EI per unit cost, but also consider the absolute EI
    
    # For Pareto frontier approach:
    # Objective 1: Minimize predicted value (i.e., maximize -mu or minimize mu)
    # Objective 2: Minimize cost
    
    # We'll use non-dominated sorting on (mu, cost) - lower is better for both
    # Then assign scores based on dominance rank and efficiency
    
    # Efficient approach: use a simplified Pareto rank
    # Sort by cost, then for each cost level, keep track of best mu seen
    
    # Actually, let's use a proper non-dominated sort approximation
    # For 2000 points, we can do an O(n^2) dominated check or use a faster method
    
    # Faster approach: 
    # 1. Normalize mu and cost to [0,1] range
    # 2. Compute a composite score that emphasizes the Pareto frontier
    
    mu_min = np.min(mu)
    mu_max = np.max(mu)
    cost_min = np.min(cost)
    cost_max = np.max(cost)
    
    mu_range = mu_max - mu_min + 1e-9
    cost_range = cost_max - cost_min + 1e-9
    
    mu_norm = (mu - mu_min) / mu_range  # 0 = best (lowest), 1 = worst
    cost_norm = (cost - cost_min) / cost_range  # 0 = cheapest, 1 = most expensive
    
    # Pareto dominance: point A dominates B if A is better in at least one objective
    # and not worse in any. Lower mu and lower cost are better.
    
    # For efficiency, let's use a rank-based approach:
    # For each point, count how many other points dominate it
    # Points with 0 dominators are on the Pareto front (rank 1)
    # Points dominated only by rank-1 points are rank 2, etc.
    
    # O(n^2) non-dominated sort for n=2000 is 4M operations, should be fast
    
    dominated_count = np.zeros(n, dtype=np.int32)
    rank = np.zeros(n, dtype=np.int32)
    
    # Build dominance structure
    # A dominates B if mu[a] <= mu[b] and cost[a] <= cost[b] and (mu[a] < mu[b] or cost[a] < cost[b])
    
    # For speed, sort by mu first, then process
    order_mu = np.argsort(mu)
    
    # For each point in order of increasing mu, count how many with lower/equal mu 
    # also have lower/equal cost
    # This is a 2D dominance counting problem
    
    # Simpler: for n=2000, do pairwise comparison
    # But let's be smarter: sort by mu, then for each point, the dominators 
    # are those with mu <= mu[i] and cost <= cost[i], excluding itself
    
    # Process in order of increasing mu
    # Maintain a running minimum of cost seen so far
    # If cost[i] >= min_cost_so_far, then point i is dominated by some earlier point
    # But we need exact count for proper ranking...
    
    # Let's just do the O(n^2) since n=2000 is manageable
    # Actually, let's use a vectorized approach for speed
    
    # For each point i, count j such that mu[j] <= mu[i] and cost[j] <= cost[i] and (mu[j] < mu[i] or cost[j] < cost[i])
    
    # Vectorized: for each i, this is sum over j of indicator
    # But that's O(n^2) memory if done naively
    
    # Let's do it in chunks or use a different approach
    
    # Alternative: assign score based on a weighted combination that approximates Pareto efficiency
    # score = -mu_norm * w1 - cost_norm * w2, but with adaptive weights based on local density
    
    # Better: use the concept of "marginal improvement per cost"
    # For points near the frontier, compute how much mu decreases per unit cost
    
    # Let's use a hybrid: 
    # 1. Compute EI/cost as baseline
    # 2. Boost points that are on or near the Pareto front
    
    # Compute baseline: EI per cost
    ei_per_cost = ei / np.maximum(cost, 1e-9)
    
    # Compute Pareto front membership
    # A point is on the Pareto front if no other point has both lower mu and lower cost
    # Efficient check: sort by mu, track running min of cost
    
    sorted_idx = np.argsort(mu)
    sorted_mu = mu[sorted_idx]
    sorted_cost = cost[sorted_idx]
    
    # Running min of cost from the right (i.e., among points with mu >= current)
    # Actually, we want: for point i, is there any point j with mu[j] <= mu[i] and cost[j] <= cost[i]?
    # Sort by mu ascending. For each position in sorted order, if cost is >= min cost so far, it's dominated
    
    running_min_cost = np.inf
    is_pareto = np.zeros(n, dtype=bool)
    
    for idx in sorted_idx:
        if cost[idx] <= running_min_cost + 1e-12:
            is_pareto[idx] = True
            running_min_cost = cost[idx]
    
    # Now, for non-Pareto points, we want to score them lower
    # For Pareto points, we want to emphasize those with best EI/cost
    
    # Normalize ei_per_cost to [0, 1]
    eipc_min = np.min(ei_per_cost)
    eipc_max = np.max(ei_per_cost)
    eipc_range = eipc_max - eipc_min + 1e-9
    eipc_norm = (ei_per_cost - eipc_min) / eipc_range
    
    # Score: Pareto points get high base score + eipc_norm boost
    # Non-Pareto points get lower score
    
    # Compute a "distance to Pareto front" proxy
    # For each point, the gap in mu to the best mu achievable at its cost level
    
    # Simpler: score = is_pareto * (0.5 + 0.5 * eipc_norm) + (1 - is_pareto) * eipc_norm * 0.3
    
    scores = np.where(is_pareto, 0.5 + 0.5 * eipc_norm, 0.3 * eipc_norm)
    
    # Add a small exploration component based on sigma
    # Points with higher uncertainty get a small boost
    sigma_max = np.max(sigma_safe) + 1e-9
    sigma_norm = sigma_safe / sigma_max
    scores += 0.1 * sigma_norm
    
    # Ensure scores are finite and positive
    scores = np.maximum(scores, 1e-6)
    
    # Normalize to [0, 1] for consistency
    s_min = np.min(scores)
    s_max = np.max(scores)
    if s_max - s_min > 1e-12:
        scores = (scores - s_min) / (s_max - s_min)
    else:
        scores = np.ones_like(scores)
    
    return scores
# EVOLVE-BLOCK-END
