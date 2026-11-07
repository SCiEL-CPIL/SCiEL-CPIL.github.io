"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Rank-based Pareto front selection for cost-aware Bayesian optimization.
    
    Computes expected improvement (EI) for each candidate, then identifies the
    non-dominated set in the (EI, cost) space. Candidates on the Pareto front
    receive high scores proportional to their rank within the front, while
    dominated candidates receive lower scores based on their distance to the front.
    """
    # Compute Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z) where z = (f_best - mu) / sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is non-negative (clip small negative values from numerical issues)
    ei = np.maximum(ei, 0.0)
    
    n = len(ei)
    
    # Find non-dominated points in (ei, cost) space.
    # A point is dominated if there exists another point with >= ei and <= cost,
    # with at least one strict inequality.
    # We want to keep points that are NOT dominated.
    
    # Efficient approach: 
    # 1. Sort by ei descending. For ties, sort by cost ascending.
    # 2. Traverse and keep track of minimum cost seen so far.
    #    A point is non-dominated if its cost is strictly less than the minimum cost 
    #    seen among points with strictly higher ei, OR if it's the first with that ei level.
    
    # Sort indices by ei descending, then cost ascending
    order = np.lexsort((cost, -ei))  # primary: -ei (so ei desc), secondary: cost asc
    
    ei_sorted = ei[order]
    cost_sorted = cost[order]
    
    # Find non-dominated in sorted order
    # Process groups of equal ei. Within a group, only the lowest cost point can be
    # non-dominated (others are dominated by it). Between groups, a point is non-dominated
    # if its cost is less than the minimum cost of all previous groups.
    
    non_dominated_mask = np.zeros(n, dtype=bool)
    
    min_cost_so_far = np.inf
    i = 0
    while i < n:
        # Find group of equal ei values
        j = i + 1
        while j < n and ei_sorted[j] == ei_sorted[i]:
            j += 1
        # Group is [i, j)
        # Within this group, the point with minimum cost is the only candidate for non-dominated
        min_cost_in_group = np.min(cost_sorted[i:j])
        idx_min_cost = np.argmin(cost_sorted[i:j]) + i
        
        if min_cost_in_group < min_cost_so_far:
            # This point is non-dominated
            non_dominated_mask[order[idx_min_cost]] = True
            min_cost_so_far = min_cost_in_group
        # Else, even the best in this group is dominated by a previous point
        i = j
    
    # Now assign scores using rank-based approach
    # Points on the Pareto front get high scores, others get lower scores.
    
    # Create scores array
    scores = np.zeros(n)
    
    # Identify front points
    front_indices = np.where(non_dominated_mask)[0]
    if len(front_indices) > 0:
        # Among front points, rank them by a combined criterion
        # We want to reward high EI and low cost
        front_ei = ei[front_indices]
        front_cost = cost[front_indices]
        
        # Rank front points: primary by ei descending, secondary by cost ascending
        front_order = np.lexsort((front_cost, -front_ei))
        
        # Assign ranks: 0 is best, len-1 is worst among front
        n_front = len(front_indices)
        for rank, idx_in_front in enumerate(front_order):
            idx = front_indices[idx_in_front]
            # Score proportional to (1 - rank/n_front), so best front point gets highest score
            scores[idx] = 1.0 - (rank / max(n_front - 1, 1))
        
        # Scale front scores to be clearly above non-front scores
        scores[front_indices] = 1.0 + scores[front_indices]  # range [1, 2]
    
    # For non-front points, assign scores based on how dominated they are
    non_front_indices = np.where(~non_dominated_mask)[0]
    if len(non_front_indices) > 0:
        # For each non-front point, compute a dominance penalty
        max_front_ei = np.max(ei[front_indices]) if len(front_indices) > 0 else np.max(ei)
        max_cost = np.max(cost)
        
        # Vectorized computation for non-front points
        ei_gaps = np.maximum(0.0, max_front_ei - ei[non_front_indices])
        ei_scale = max(max_front_ei, 1e-9)
        penalties = ei_gaps / ei_scale
        cost_penalties = cost[non_front_indices] / max(max_cost, 1e-9)
        
        # Combined score in [0, 1)
        score_vals = np.maximum(0.0, 1.0 - 0.5 * penalties - 0.5 * cost_penalties)
        scores[non_front_indices] = score_vals * 0.9  # Keep below 1.0 to be clearly below front scores
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
