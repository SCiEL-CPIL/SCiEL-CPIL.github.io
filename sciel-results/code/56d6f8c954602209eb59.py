"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard: stop after 120s wall-clock to stay under 150s limit
_START_TIME = time.time()
_DEADLINE = 120.0  # seconds, safety margin under 150s limit


def _elapsed():
    return time.time() - _START_TIME


def _compute_lambda(weights, values, capacity, n_iters=30):
    """
    Find the Lagrangian multiplier lambda such that the LP relaxation
    (take all items with value/weight > lambda, fractionally take the
    one equal to lambda) approximately exhausts the capacity.
    Uses bisection on lambda.
    """
    densities = values / weights
    min_d = np.min(densities)
    max_d = np.max(densities)
    
    if max_d <= min_d:
        return max_d
    
    # Bisection: find lambda where total weight taken ≈ capacity
    lo = min_d
    hi = max_d
    
    for _ in range(n_iters):
        mid = (lo + hi) / 2.0
        # Items with density > mid are taken fully
        # Item with density == mid is taken fractionally
        # Items with density < mid are not taken
        full_mask = densities > mid
        frac_mask = densities == mid
        total_w = np.sum(weights[full_mask])
        if np.any(frac_mask):
            # Take fraction of the boundary item
            remaining = capacity - total_w
            if remaining > 0:
                frac_idx = np.where(frac_mask)[0][0]
                frac_w = weights[frac_idx]
                total_w += min(remaining, frac_w)
        if total_w > capacity:
            hi = mid
        else:
            lo = mid
    
    return (lo + hi) / 2.0


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    """
    Select next item using Lagrangian relaxation:
    - Compute shadow price lambda via bisection on the LP relaxation.
    - Score each feasible item by (value - lambda * weight).
    - Pick the item with the highest Lagrangian score.
    - Tie-break by density, then value.
    """
    # Deadline check
    if _elapsed() > _DEADLINE * 0.95:
        return -1
    
    n = len(remaining_weights)
    if n == 0:
        return -1
    
    # Filter to feasible items only
    feasible_mask = remaining_weights <= remaining_capacity
    feasible_indices = np.where(feasible_mask)[0]
    
    if len(feasible_indices) == 0:
        return -1
    
    fw = remaining_weights[feasible_mask]
    fv = remaining_values[feasible_mask]
    n_feasible = len(fw)
    
    if n_feasible == 1:
        return int(feasible_indices[0])
    
    # Compute lambda via LP relaxation bisection
    lam = _compute_lambda(fw, fv, remaining_capacity, n_iters=25)
    
    # Lagrangian score: value - lambda * weight
    lag_scores = fv - lam * fw
    
    # For items with density > lambda, lag_score > 0
    # For items with density < lambda, lag_score < 0
    # The best item in the LP is the one with density closest to lambda from above
    # (or the top-density item if all are above lambda)
    
    # Strategy: among items with positive Lagrangian score (density > lambda),
    # pick the one with highest lag_score. If none, pick highest density.
    positive_mask = lag_scores > 0
    
    if np.any(positive_mask):
        # Among positive-score items, pick highest lag_score
        # Break ties by density, then value
        pos_scores = lag_scores.copy()
        pos_scores[~positive_mask] = -np.inf
        best_local = int(np.argmax(pos_scores))
    else:
        # All items have density <= lambda; pick highest density
        densities = fv / fw
        best_local = int(np.argmax(densities))
    
    # Secondary refinement: if the best item by Lagrangian has very low score,
    # consider a lookahead: simulate taking it and estimate remaining fill
    # For speed, just use the Lagrangian pick directly
    
    return int(feasible_indices[best_local])

# EVOLVE-BLOCK-END
