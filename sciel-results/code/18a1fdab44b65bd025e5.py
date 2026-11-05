"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard: 150s wall-clock limit, use 145s as safety margin
_DEADLINE = None
_START_TIME = None


def _ensure_deadline():
    global _DEADLINE, _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
        _DEADLINE = _START_TIME + 145.0
    return _DEADLINE


def _time_exceeded():
    global _DEADLINE
    if _DEADLINE is None:
        _ensure_deadline()
    return time.time() >= _DEADLINE


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    """
    Lagrangian relaxation heuristic for 0/1 knapsack.
    
    Selects the item maximizing v_i - lambda * w_i, where lambda is tuned
    via binary search to match the remaining capacity (fractional knapsack
    relaxation). This avoids local density traps by considering the global
    trade-off between value and weight.
    """
    global _DEADLINE, _START_TIME
    deadline = _ensure_deadline()
    
    n = len(remaining_weights)
    if n == 0:
        return -1
    
    # If no capacity left, stop
    if remaining_capacity <= 1e-12:
        return -1
    
    rw = np.asarray(remaining_weights, dtype=np.float64)
    rv = np.asarray(remaining_values, dtype=np.float64)
    rc = float(remaining_capacity)
    
    # Check for items with zero or negative weight (take them all, pick first)
    zero_weight_mask = rw <= 1e-12
    if np.any(zero_weight_mask):
        # Pick the one with highest value among zero-weight items
        zero_indices = np.where(zero_weight_mask)[0]
        return int(zero_indices[np.argmax(rv[zero_indices])])
    
    # Binary search for lambda (Lagrangian multiplier) such that the sum of
    # weights of items with positive (v_i - lambda * w_i) approximately equals rc.
    # 
    # The fractional knapsack optimal solution: sort by v_i/w_i descending,
    # take all items until capacity is filled. The lambda at the breakpoint
    # is v_j/w_j for the item that partially fills the knapsack.
    #
    # We find lambda such that sum(w_i * 1[v_i - lambda*w_i > 0]) <= rc
    # and sum(w_i * 1[v_i - lambda*w_i >= 0]) >= rc (approximately).
    
    # Compute value-to-weight ratios
    ratios = rv / rw  # v_i / w_i
    
    # The lambda range: [0, max_ratio]
    # At lambda = 0, all items are "selected" (positive reduced cost)
    # At lambda = max_ratio, only the best ratio item might be selected
    
    lo = 0.0
    hi = float(np.max(ratios))
    
    # Binary search for the critical lambda
    # We want the largest lambda such that the total weight of items with
    # v_i - lambda * w_i > 0 is >= rc
    # Actually, we want to find the lambda where the "marginal" item has
    # v_i/w_i = lambda, and items with higher ratio are taken.
    
    # More precisely: find lambda such that
    #   sum_{i: v_i/w_i > lambda} w_i <= rc <= sum_{i: v_i/w_i >= lambda} w_i
    # This is the fractional knapsack breakpoint.
    
    # Sort by ratio descending
    order = np.argsort(-ratios)
    sorted_w = rw[order]
    sorted_r = rv[order]
    sorted_ratio = ratios[order]
    
    # Find the breakpoint
    cum_w = np.cumsum(sorted_w)
    
    # Find where cumulative weight crosses rc
    cross_idx = np.searchsorted(cum_w, rc, side='left')
    
    if cross_idx == 0:
        # Even the best item doesn't fit? Shouldn't happen if we filtered feasible
        # But just in case, pick the best ratio item
        return int(order[0])
    
    if cross_idx >= n:
        # All items fit, pick the one with highest ratio (first in sorted order)
        return int(order[0])
    
    # The breakpoint lambda is the ratio at index cross_idx (the item that
    # partially fills the knapsack in fractional solution)
    lambda_star = float(sorted_ratio[cross_idx])
    
    # Now, select the item that maximizes v_i - lambda_star * w_i
    # Among items with v_i - lambda_star * w_i > 0 (i.e., ratio > lambda_star)
    # or the marginal item itself.
    
    reduced_costs = rv - lambda_star * rw
    
    # Items with strictly positive reduced cost are "clearly good"
    # The marginal item (at the breakpoint) has reduced cost ~ 0
    
    # Find the best item by reduced cost
    best_idx = int(np.argmax(reduced_costs))
    
    # But we should prefer items with higher ratio if reduced costs are close
    # Let's add a small tie-breaking bonus for higher ratio
    # Actually, let's just use the reduced cost, but with a slight preference
    # for items that fit well within remaining capacity
    
    # Alternative: among items with reduced cost > 0, pick the one with
    # highest reduced cost. If none, pick the marginal item.
    
    positive_mask = reduced_costs > 1e-9
    if np.any(positive_mask):
        candidates = np.where(positive_mask)[0]
        # Among these, pick highest reduced cost
        best_among_pos = candidates[np.argmax(reduced_costs[candidates])]
        return int(best_among_pos)
    
    # If no positive reduced cost, pick the marginal item (highest ratio that fits)
    # This should be the item at cross_idx in sorted order
    return int(order[cross_idx])


# EVOLVE-BLOCK-END
