"""Cost-aware acquisition function for Bayesian optimization (styblinski tang 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Pareto-frontier based scoring.
    
    Parameters:
    -----------
    mu : array of shape (2000,) - GP mean predictions
    sigma : array of shape (2000,) - GP standard deviations
    f_best : float - best observed value so far
    cost : array of shape (2000,) - cost of evaluating each candidate
    
    Returns:
    --------
    scores : array of shape (2000,) - acquisition scores (higher = better)
    """
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 300s limit
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure finite values
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-10)
    sigma = np.maximum(sigma, 1e-10)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    cost = np.maximum(cost, 1e-10)
    
    n = len(mu)
    
    # Compute the two key quantities for Pareto-based scoring:
    # 1. Predicted value (lower is better): use a combination of mu and uncertainty
    # 2. Evaluation cost (lower is better)
    
    # For predicted value, use expected minimum considering uncertainty
    # This is the lower confidence bound (LCB) for minimization
    # LCB = mu - beta * sigma (lower values are better for minimization)
    
    # Adaptive beta based on uncertainty
    sigma_mean = np.mean(sigma)
    if sigma_mean > 1.0:
        beta = 2.5
    elif sigma_mean > 0.1:
        beta = 1.5
    else:
        beta = 0.8
    
    # Lower confidence bound (predicted best value at each point)
    lcb = mu - beta * sigma
    
    # Expected improvement for reference
    sigma_safe = np.maximum(sigma, 1e-10)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Pareto-frontier based scoring:
    # We want to find points that are non-dominated in (lcb, cost) space
    # where lower lcb and lower cost are both better
    
    # Step 1: Normalize both dimensions to [0, 1] for fair comparison
    lcb_min, lcb_max = np.min(lcb), np.max(lcb)
    cost_min, cost_max = np.min(cost), np.max(cost)
    
    # Handle degenerate cases
    if lcb_max - lcb_min < 1e-10:
        lcb_norm = np.zeros(n)
    else:
        lcb_norm = (lcb - lcb_min) / (lcb_max - lcb_min)
    
    if cost_max - cost_min < 1e-10:
        cost_norm = np.zeros(n)
    else:
        cost_norm = (cost - cost_min) / (cost_max - cost_min)
    
    # Step 2: Compute Pareto ranks using non-dominated sorting
    # A point is non-dominated if no other point has both lower lcb AND lower cost
    
    # For efficiency, use a vectorized approach:
    # A point i is dominated by point j if lcb[j] <= lcb[i] AND cost[j] <= cost[i]
    # and at least one is strictly less
    
    # Compute dominance count for each point
    # point i is dominated if there exists j with lcb[j] <= lcb[i] and cost[j] <= cost[i]
    # and (lcb[j] < lcb[i] or cost[j] < cost[i])
    
    # Efficient approach: sort by lcb, then for each point, check if any earlier point
    # has cost <= current cost
    
    # Sort indices by lcb (ascending)
    sort_idx = np.argsort(lcb)
    lcb_sorted = lcb[sort_idx]
    cost_sorted = cost[sort_idx]
    
    # For each point in sorted order, compute the minimum cost seen so far (excluding itself)
    # If min_cost_so_far < current cost, then current point is dominated
    
    # Use a running minimum from the left
    min_cost_before = np.full(n, np.inf)
    for i in range(1, n):
        min_cost_before[i] = min(min_cost_before[i-1], cost_sorted[i-1])
    
    # A point is non-dominated if no earlier point (with lower or equal lcb) has lower or equal cost
    # More precisely: non-dominated if min_cost_before[i] >= cost_sorted[i] 
    # (no point with strictly lower lcb has cost <= current cost)
    # But we also need to handle ties in lcb
    
    # Better approach: for each point, count how many points dominate it
    # Point j dominates point i if lcb[j] < lcb[i] and cost[j] <= cost[i], OR lcb[j] <= lcb[i] and cost[j] < cost[i]
    
    # Vectorized dominance count:
    # For each pair (i, j), j dominates i if (lcb[j] < lcb[i] and cost[j] <= cost[i]) or 
    # (lcb[j] <= lcb[i] and cost[j] < cost[i])
    
    # This is O(n^2) which is too slow for n=2000 potentially, but let's try a smarter approach
    
    # Use the fact that if we sort by lcb, then for point at position i in sorted order,
    # it is dominated if there exists j < i with cost_sorted[j] <= cost_sorted[i]
    # (since lcb_sorted[j] <= lcb_sorted[i] for j < i)
    # But we need strict inequality in at least one dimension
    
    # Refined: point at sorted position i is dominated if:
    # - There exists j < i with lcb_sorted[j] < lcb_sorted[i] and cost_sorted[j] <= cost_sorted[i]
    # - OR there exists j <= i with lcb_sorted[j] <= lcb_sorted[i] and cost_sorted[j] < cost_sorted[i]
    
    # Simplified: compute for each point whether it's on the Pareto frontier
    # A point is on the Pareto frontier if it's not dominated by any other point
    
    # Efficient method: 
    # 1. Sort by lcb ascending
    # 2. Scan from left to right, maintaining the minimum cost seen so far
    # 3. A point is non-dominated if its cost is strictly less than all previous costs
    #    (because any previous point has lcb <= current lcb, so if previous cost < current cost, 
    #     the current point is dominated)
    
    is_non_dominated = np.zeros(n, dtype=bool)
    min_cost_so_far = np.inf
    for i in range(n):
        idx = sort_idx[i]
        if cost_sorted[i] < min_cost_so_far:
            is_non_dominated[idx] = True
            min_cost_so_far = cost_sorted[i]
    
    # Step 3: Assign scores based on Pareto rank
    # Points on the Pareto frontier get high scores
    # Points not on the frontier get scores based on their distance to the frontier
    
    # For non-frontier points, compute a penalty based on how much they are dominated
    # Use the normalized sum of (lcb_norm + cost_norm) as a fallback score
    
    # Base score: combination of normalized lcb and cost
    # Lower normalized values are better, so we invert
    base_score = -(lcb_norm + cost_norm) / 2.0
    
    # Boost for non-dominated points
    # The boost should reflect how much better they are on the frontier
    
    # Compute a "frontier score" for non-dominated points based on their position
    # Points with lower lcb and lower cost should get higher scores
    
    # For non-dominated points, score = -lcb_norm (since cost is implicitly minimized on the frontier)
    # Actually, use a weighted combination that emphasizes cost-efficiency
    
    # Cost-efficiency: marginal reduction in lcb per unit cost
    # For each point, estimate how much lcb can be reduced by moving to a lower-cost point
    
    # Simpler approach: score = -lcb_norm * (1 + cost_efficiency_bonus)
    # where cost_efficiency_bonus rewards points that achieve low lcb at low cost
    
    # Compute cost efficiency: for each point, how much lcb is reduced relative to the 
    # maximum lcb achievable at this cost level
    
    # For each cost level, find the best (lowest) lcb
    # This is essentially the Pareto frontier itself
    
    # Score for non-dominated points:
    # Higher score for points that are more "extreme" on the frontier
    # Use -lcb_norm as the score, since on the frontier, lower lcb means lower cost too
    
    scores = base_score.copy()
    
    # Boost non-dominated points
    if np.any(is_non_dominated):
        # Among non-dominated points, rank by lcb_norm (lower is better)
        nd_indices = np.where(is_non_dominated)[0]
        nd_lcb_norm = lcb_norm[nd_indices]
        
        # Normalize lcb_norm among non-dominated points to [0, 1]
        if len(nd_lcb_norm) > 1:
            nd_lcb_min = np.min(nd_lcb_norm)
            nd_lcb_max = np.max(nd_lcb_norm)
            if nd_lcb_max - nd_lcb_min > 1e-10:
                nd_rank = (nd_lcb_norm - nd_lcb_min) / (nd_lcb_max - nd_lcb_min)
            else:
                nd_rank = np.zeros(len(nd_lcb_norm))
        else:
            nd_rank = np.zeros(1)
        
        # Higher score for lower nd_rank (better lcb)
        # Score = base_score + boost * (1 - nd_rank)
        boost = 1.0
        scores[nd_indices] = base_score[nd_indices] + boost * (1.0 - nd_rank)
    
    # Check deadline
    if time.time() > deadline:
        return np.where(np.isfinite(scores), scores, 0.0)
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Final normalization to ensure reasonable scale
    score_min = np.min(scores)
    score_max = np.max(scores)
    if score_max - score_min > 1e-10:
        scores = (scores - score_min) / (score_max - score_min)
    
    return scores
# EVOLVE-BLOCK-END
