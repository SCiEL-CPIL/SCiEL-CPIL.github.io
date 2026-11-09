"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 115.0  # 5s safety margin before 120s limit


def _elapsed_ok():
    return (time.time() - _START_TIME) < _DEADLINE


def _greedy_fill_value(capacity, weights, values, exclude_idx=None):
    """Return total value from greedy density fill, optionally excluding one item."""
    n = len(weights)
    if n == 0 or capacity <= 0:
        return 0.0
    
    # Create indices excluding the excluded item
    if exclude_idx is not None:
        mask = np.arange(n) != exclude_idx
        idxs = np.where(mask)[0]
        w = weights[idxs]
        v = values[idxs]
    else:
        idxs = np.arange(n)
        w = weights
        v = values
    
    if len(w) == 0:
        return 0.0
    
    # Sort by density descending
    densities = v / w
    order = np.argsort(-densities)
    
    cap = capacity
    val = 0.0
    for i in order:
        wi = w[i]
        if wi <= cap:
            cap -= wi
            val += v[i]
        if cap <= 0:
            break
    return val


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1

    # If time is running out, fall back to simple density greedy
    if not _elapsed_ok():
        return int(np.argmax(remaining_values / remaining_weights))

    # UCB-style bandit approach:
    # For each feasible item, compute:
    #   exploitation = value density (value/weight)
    #   exploration = uncertainty based on weight synergy with remaining capacity
    #   UCB score = exploitation + exploration_term
    
    weights = np.asarray(remaining_weights, dtype=np.float64)
    values = np.asarray(remaining_values, dtype=np.float64)
    capacity = float(remaining_capacity)
    
    # Feasible items only
    feasible_mask = weights <= capacity
    feasible_indices = np.where(feasible_mask)[0]
    
    if len(feasible_indices) == 0:
        return -1
    
    # Compute base density for all items
    densities = values / weights
    
    # For UCB: balance between high-density items and items that might 
    # unlock better combinations due to weight fit
    
    # Exploitation: density
    # Exploration: items with weight close to remaining capacity (good fit) 
    #               or items that leave "nice" remaining capacity for future items
    
    best_score = -np.inf
    best_idx = feasible_indices[0]
    
    # Pre-compute: what's the greedy fill value if we DON'T pick each item?
    # This is expensive, so we'll use a faster proxy for exploration
    
    # Exploration term: how well does this item's weight complement the capacity?
    # Items that leave a remaining capacity that can be well-utilized are better
    # Proxy: 1 - (remaining_capacity - weight) / remaining_capacity
    # This rewards items that use more of the current capacity
    
    # Also consider: items with weight that divides remaining capacity nicely
    # (i.e., remaining_capacity / weight is close to an integer)
    
    for i in feasible_indices:
        wi = weights[i]
        vi = values[i]
        di = densities[i]
        
        # Exploitation component: density
        exploit = di
        
        # Exploration component: 
        # 1. Fit bonus: how much of capacity this item uses
        fit_bonus = wi / capacity
        
        # 2. Synergy bonus: does the remaining capacity after picking this item
        #    allow for good subsequent picks?
        #    Proxy: if remaining capacity is small, it's less risky
        #    If remaining capacity is large, we have more flexibility
        rem_cap = capacity - wi
        if rem_cap < 1e-9:
            synergy = 1.0  # Perfect fit
        else:
            # Check if there are items that fit well in the remaining capacity
            # Fast proxy: count items with weight <= rem_cap and weight > rem_cap/2
            # (items that would use a significant portion of remaining capacity)
            if rem_cap > 0:
                good_fits = np.sum((weights <= rem_cap) & (weights > rem_cap * 0.5))
                synergy = min(1.0, good_fits / max(1, n * 0.1))
            else:
                synergy = 0.0
        
        # UCB score: combine exploitation and exploration
        # Use a weighted combination that adapts based on how many items remain
        # Early on: more exploration; later: more exploitation
        frac_remaining = n / 100.0  # Approximate fraction of items remaining
        explore_weight = 0.3 * frac_remaining + 0.1
        
        ucb_score = exploit * (1 - explore_weight) + (fit_bonus * 0.5 + synergy * 0.5) * explore_weight
        
        # Normalize density to be comparable with fit/synergy terms
        # Scale density by a factor to make it comparable
        ucb_score = di * 10.0 + fit_bonus * 2.0 + synergy * 1.5
        
        if ucb_score > best_score:
            best_score = ucb_score
            best_idx = i
        
        # Time check
        if not _elapsed_ok():
            break
    
    return int(best_idx)
# EVOLVE-BLOCK-END
