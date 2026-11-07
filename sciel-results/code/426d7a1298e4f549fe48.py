"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining EI, UCB exploration,
    and a non-linear cost penalty with rank-based Pareto front scoring.
    
    Strategy:
    1. Compute EI (expected improvement) per candidate.
    2. Compute UCB (upper confidence bound) for exploration.
    3. Combine into a composite score with non-linear cost penalty.
    4. Apply rank-based Pareto front selection to enforce trade-off.
    
    Returns 2000 finite scores (higher = evaluate next).
    """
    # Deadline guard
    t0 = time.time()
    DEADLINE = 330.0 - 5.0  # 5s safety margin
    
    n = len(mu)
    
    # Handle edge cases
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # UCB for exploration (beta controls exploration-exploitation trade-off)
    beta = 2.0
    ucb = mu + beta * sigma_safe
    
    # Combine EI and UCB: use EI as primary, UCB as auxiliary
    # Normalize both to [0, 1] range for fair combination
    ei_min, ei_max = np.min(ei), np.max(ei)
    ucb_min, ucb_max = np.min(ucb), np.max(ucb)
    
    ei_norm = (ei - ei_min) / (ei_max - ei_min + 1e-9)
    ucb_norm = (ucb - ucb_min) / (ucb_max - ucb_min + 1e-9)
    
    # Composite score: weighted combination
    w_ei = 0.7
    w_ucb = 0.3
    composite = w_ei * ei_norm + w_ucb * ucb_norm
    
    # Non-linear cost penalty: cost^gamma with offset
    # Lower cost (cheaper evaluations) should be favored
    # cost is in [1, 2] range (1 + mean(x scaled to [0,1]))
    gamma = 1.5
    offset = 0.6
    cost_penalty = np.power(cost + offset, gamma)
    
    # Score = composite / cost_penalty
    # This favors candidates with high improvement potential AND low cost
    scores = composite / (cost_penalty + 1e-9)
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Rank-based Pareto front selection
    # Find non-dominated set based on (ei, -cost) trade-off
    # A candidate is dominated if another has >= EI and <= cost
    
    # Build Pareto front: sort by EI descending, keep track of min cost
    ei_desc_idx = np.argsort(-ei)
    
    # Assign Pareto ranks
    pareto_rank = np.zeros(n)
    min_cost_so_far = np.inf
    rank = 1
    for i in ei_desc_idx:
        if cost[i] < min_cost_so_far:
            pareto_rank[i] = rank
            min_cost_so_far = cost[i]
            rank += 1
        else:
            pareto_rank[i] = rank  # dominated, gets current or higher rank
    
    # Actually, let's do proper non-dominated sorting
    # Non-dominated: no other point has both >= EI and <= cost
    non_dominated = np.ones(n, dtype=bool)
    
    # Vectorized check: for each point, check if any other dominates it
    # Point j dominates point i if ei[j] >= ei[i] and cost[j] <= cost[i] and (ei[j] > ei[i] or cost[j] < cost[i])
    
    # For efficiency with 2000 points, use a simpler approach
    # Sort by EI descending, then scan for decreasing cost
    ei_order = np.argsort(-ei)
    
    # Find Pareto front indices
    is_pareto = np.zeros(n, dtype=bool)
    min_cost = np.inf
    for idx in ei_order:
        if cost[idx] < min_cost:
            is_pareto[idx] = True
            min_cost = cost[idx]
    
    # Assign scores: Pareto front points get boosted scores
    # Non-Pareto points get their base score
    if np.any(is_pareto):
        pareto_scores = scores[is_pareto]
        # Boost Pareto front points
        boost = 1.5
        scores[is_pareto] = pareto_scores * boost
    
    # Final: ensure all scores are finite and non-negative
    scores = np.where(np.isfinite(scores), scores, 0.0)
    scores = np.maximum(scores, 0.0)
    
    # Check deadline
    if time.time() - t0 > DEADLINE:
        # Return best result found so far (already computed)
        pass
    
    return scores
# EVOLVE-BLOCK-END
