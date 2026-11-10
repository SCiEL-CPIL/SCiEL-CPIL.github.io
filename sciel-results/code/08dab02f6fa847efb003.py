"""Improved 0/1 knapsack heuristic with fractional knapsack surrogate lookahead."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 140.0  # seconds; safety margin before 150s kill
_START = None


def _fractional_knapsack_value(weights, values, capacity, exclude_idx=None):
    """Compute fractional knapsack value (upper bound), optionally excluding one index."""
    if capacity <= 0:
        return 0.0
    n = len(weights)
    if n == 0:
        return 0.0
    if exclude_idx is not None:
        mask = np.ones(n, dtype=bool)
        mask[exclude_idx] = False
        w = weights[mask]
        v = values[mask]
    else:
        w = weights
        v = values
    if len(w) == 0:
        return 0.0
    density = v / w
    order = np.argsort(density)[::-1]
    w_sorted = w[order]
    v_sorted = v[order]
    total_val = 0.0
    remaining_cap = capacity
    for i in range(len(w_sorted)):
        if remaining_cap <= 0:
            break
        if w_sorted[i] <= remaining_cap:
            total_val += v_sorted[i]
            remaining_cap -= w_sorted[i]
        else:
            total_val += v_sorted[i] * (remaining_cap / w_sorted[i])
            remaining_cap = 0
    return float(total_val)


def _greedy_fill_value(weights, values, capacity, exclude_idx=None):
    """Compute total value from greedy density fill (0/1), optionally excluding one index."""
    if capacity <= 0:
        return 0.0
    n = len(weights)
    if n == 0:
        return 0.0
    if exclude_idx is not None:
        mask = np.ones(n, dtype=bool)
        mask[exclude_idx] = False
        w = weights[mask]
        v = values[mask]
    else:
        w = weights
        v = values
    if len(w) == 0:
        return 0.0
    density = v / w
    order = np.argsort(density)[::-1]
    w_sorted = w[order]
    v_sorted = v[order]
    cum_w = np.cumsum(w_sorted)
    cum_v = np.cumsum(v_sorted)
    idx = np.searchsorted(cum_w, capacity, side='right') - 1
    if idx < 0:
        return 0.0
    return float(cum_v[idx])


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    """
    Select next item for 0/1 knapsack using fractional knapsack surrogate lookahead.
    """
    global _START
    if _START is None:
        _START = time.time()
    elapsed = time.time() - _START
    if elapsed > _DEADLINE:
        return int(np.argmax(remaining_values / remaining_weights))

    n = len(remaining_weights)
    if n == 0:
        return -1
    if remaining_capacity <= 0:
        return -1

    if n == 1:
        if remaining_weights[0] <= remaining_capacity:
            return 0
        return -1

    densities = remaining_values / remaining_weights

    # Use fractional knapsack to identify which items are in the continuous optimum
    # Items with density above the threshold are fully included in the fractional solution
    frac_val = _fractional_knapsack_value(remaining_weights, remaining_values, remaining_capacity)
    
    # Find the threshold density: items with density > threshold are fully in the fractional solution
    # We can approximate this by finding how many items fit fully
    order = np.argsort(densities)[::-1]
    w_sorted = remaining_weights[order]
    v_sorted = remaining_values[order]
    cum_w = np.cumsum(w_sorted)
    # Number of items that fit fully in fractional solution
    num_full = int(np.searchsorted(cum_w, remaining_capacity, side='right') - 1)
    if num_full < 0:
        num_full = 0
    
    # Top-k candidates: prioritize items that are in the fractional solution
    # Items 0..num_full-1 in sorted order are fully included
    # Items at position num_full might be partially included
    # Items beyond num_full are not included in fractional solution
    
    # Build candidate list: all items that fit, sorted by a combined score
    # Score: fractional inclusion bonus + density + small waste penalty
    k = min(n, 40)
    top_k_idx = np.argsort(densities)[::-1][:k]
    
    best_score = -1.0
    best_idx = top_k_idx[0]
    
    for i in top_k_idx:
        w_i = remaining_weights[i]
        v_i = remaining_values[i]
        if w_i > remaining_capacity:
            continue
        
        # Fractional knapsack value of remaining subproblem after taking i
        rem_cap = remaining_capacity - w_i
        frac_bound = _fractional_knapsack_value(remaining_weights, remaining_values, rem_cap, exclude_idx=int(i))
        
        # Greedy 0/1 fill value (actual achievable)
        greedy_val = _greedy_fill_value(remaining_weights, remaining_values, rem_cap, exclude_idx=int(i))
        
        # Combined score: blend of fractional bound (tighter) and greedy (achievable)
        # Weight towards greedy since it's what we can actually achieve
        total_val = v_i + 0.3 * frac_bound + 0.7 * greedy_val
        
        # Bonus if this item is in the fractional optimum (high density relative to threshold)
        # Items with higher density are more likely to be in the fractional solution
        density_bonus = 0.01 * densities[i] * rem_cap
        
        # Penalty for wasting capacity
        waste_penalty = 0.001 * rem_cap
        
        score = total_val + density_bonus - waste_penalty
        
        if score > best_score:
            best_score = score
            best_idx = int(i)
    
    # Check deadline again
    if time.time() - _START > _DEADLINE:
        return int(np.argmax(remaining_values / remaining_weights))
    
    return best_idx
# EVOLVE-BLOCK-END
