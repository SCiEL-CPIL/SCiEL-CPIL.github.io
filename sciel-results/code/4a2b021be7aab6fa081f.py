"""Rewritten program for ant colony bin packing heuristic design."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(demands, capacity):
    """
    Compute an item-by-item attractiveness matrix for placing item j right after
    item i in the same bin. Larger values indicate more attractive placements.
    
    New approach: Use a "tightness" heuristic that rewards pairs (i, j) where
    the combined size is close to capacity (high utilization) while still
    leaving enough room for at least one more item. This directly optimizes
    for bin utilization, which is the key factor in minimizing the number of bins.
    
    The heuristic combines:
      1. Utilization score: higher when demands[i] + demands[j] is closer to capacity
      2. Future fit score: rewards residuals that can accommodate other items
      3. Penalty for very small residuals (wasted space) and very large residuals
         (under-utilization)
    """
    demands = np.asarray(demands, dtype=np.float64)
    n = len(demands)
    
    # Pairwise sums
    pair_sum = demands[:, None] + demands[None, :]
    
    # Feasibility: pair must fit
    feasible = pair_sum <= capacity
    
    # Residual capacity
    residual = capacity - pair_sum
    
    # Utilization score: prefer high utilization (pair_sum close to capacity)
    # Normalize pair_sum to [0, 1] relative to capacity
    utilization = pair_sum / capacity
    # Score is higher when utilization is higher, but we need to be careful
    # about the residual being too small
    
    # Future fit score: how many other items could fit in the residual?
    # For each pair (i,j), count how many items k (k != i, j) could fit
    # in the residual capacity. This is expensive, so we approximate.
    
    # Approximate future fit: residual / min_demand gives an upper bound on
    # how many more items could fit. We use a smooth function.
    min_demand = np.min(demands)
    max_demand = np.max(demands)
    
    # Score based on residual: we want residual to be in a "sweet spot"
    # Too small: wasted space (< min_demand, can't fit anything else)
    # Too large: under-utilization (> capacity - min_demand, could fit bigger items)
    # Sweet spot: around min_demand to capacity - max_demand
    
    # Use a trapezoidal-like function for residual scoring
    # Low edge: 0, high edge: min_demand (linear increase)
    # Flat top: min_demand to capacity - max_demand
    # High edge: capacity - max_demand to capacity (linear decrease)
    
    low_edge = 0.0
    high_edge_start = min_demand
    high_edge_end = capacity - max_demand
    upper_edge = capacity
    
    # Residual score
    residual_score = np.zeros_like(residual)
    
    # Region 1: residual < high_edge_start (increasing)
    mask1 = (residual >= low_edge) & (residual < high_edge_start)
    residual_score[mask1] = (residual[mask1] - low_edge) / (high_edge_start - low_edge)
    
    # Region 2: high_edge_start <= residual <= high_edge_end (flat, score = 1)
    mask2 = (residual >= high_edge_start) & (residual <= high_edge_end)
    residual_score[mask2] = 1.0
    
    # Region 3: high_edge_end < residual < upper_edge (decreasing)
    mask3 = (residual > high_edge_end) & (residual < upper_edge)
    residual_score[mask3] = (upper_edge - residual[mask3]) / (upper_edge - high_edge_end)
    
    # Combine utilization and residual scores
    # Weight: 0.6 for utilization, 0.4 for residual
    heuristic = 0.6 * utilization + 0.4 * residual_score
    
    # Ensure positive values for feasible pairs
    heuristic = np.where(feasible, heuristic, 1e-9)
    
    # Clamp to ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic), heuristic, 1e-9)
    heuristic = np.maximum(heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
