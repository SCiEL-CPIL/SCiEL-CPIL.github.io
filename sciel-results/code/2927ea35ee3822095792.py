"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.monotonic()
_TIME_LIMIT = 115.0  # Safety margin before 120s limit

def _fractional_bound(capacity, weights, values, exclude_idx):
    """
    Calculate the fractional knapsack upper bound for the remaining items,
    excluding the item at 'exclude_idx'.
    """
    if capacity <= 0:
        return 0.0
    
    n = len(weights)
    if n == 0:
        return 0.0
        
    indices = [i for i in range(n) if i != exclude_idx]
    if not indices:
        return 0.0
        
    items = []
    for i in indices:
        w = weights[i]
        v = values[i]
        if w <= 0:
            continue
        items.append((v / w, w, v))
        
    if not items:
        return 0.0
        
    items.sort(key=lambda x: x[0], reverse=True)
    
    total_val = 0.0
    remaining_cap = capacity
    
    for density, w, v in items:
        if remaining_cap <= 0:
            break
        if w <= remaining_cap:
            total_val += v
            remaining_cap -= w
        else:
            total_val += v * (remaining_cap / w)
            remaining_cap = 0
            break
            
    return total_val

def _greedy_complete(capacity, weights, values, exclude_idx, mode='density'):
    """
    Complete the knapsack greedily using a specific mode.
    Returns the total value of the completed solution (excluding the value of the item at exclude_idx,
    as that is assumed to be already picked or we are just estimating the rest).
    Actually, for scoring, we want the value of the *rest* of the items we would pick.
    """
    if capacity <= 0:
        return 0.0
    
    n = len(weights)
    if n == 0:
        return 0.0
        
    indices = [i for i in range(n) if i != exclude_idx]
    if not indices:
        return 0.0
        
    items = []
    for i in indices:
        w = weights[i]
        v = values[i]
        if w <= 0:
            continue
        items.append((w, v, i))
        
    if not items:
        return 0.0
        
    if mode == 'density':
        # Sort by v/w descending
        items.sort(key=lambda x: x[1] / x[0] if x[0] > 0 else 0, reverse=True)
    elif mode == 'value':
        # Sort by value descending
        items.sort(key=lambda x: x[1], reverse=True)
    elif mode == 'weight':
        # Sort by weight ascending
        items.sort(key=lambda x: x[0])
    else:
        # Default density
        items.sort(key=lambda x: x[1] / x[0] if x[0] > 0 else 0, reverse=True)
    
    total_val = 0.0
    remaining_cap = capacity
    
    for w, v, i in items:
        if remaining_cap <= 0:
            break
        if w <= remaining_cap:
            total_val += v
            remaining_cap -= w
        # In greedy 0/1, if it doesn't fit, we skip it and try next
        # (Unlike fractional, we don't take a fraction)
        
    return total_val

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    """
    Select the next item to add to the knapsack using a lookahead strategy
    with fractional knapsack upper bound and greedy completion estimation.
    """
    # Time guard
    if time.monotonic() - _START_TIME > _TIME_LIMIT:
        return -1
        
    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1
        
    # Find all feasible items
    feasible_mask = remaining_weights <= remaining_capacity
    if not np.any(feasible_mask):
        return -1
        
    feasible_indices = np.where(feasible_mask)[0]
    
    best_score = -1.0
    best_idx = -1
    
    # We will evaluate each feasible candidate.
    # To speed up, we can pre-compute some things if n is large, but n<=100 is small.
    # However, calling _fractional_bound inside the loop is O(n log n) each time, so total O(n^2 log n).
    # For n=100, that's 100 * 100 * log(100) ~ 10000 * 7 = 70k operations. Very fast.
    
    # Convert to lists for potentially faster access in inner loops if needed,
    # but numpy indexing is fine.
    w_list = remaining_weights.tolist()
    v_list = remaining_values.tolist()
    
    for idx in feasible_indices:
        w_i = w_list[idx]
        v_i = v_list[idx]
        
        # Remaining capacity after picking idx
        new_cap = remaining_capacity - w_i
        
        # Estimate 1: Fractional Bound (Upper bound, often tight)
        completion_val_frac = _fractional_bound(new_cap, w_list, v_list, idx)
        score_frac = v_i + completion_val_frac
        
        # Estimate 2: Greedy Density Completion (Heuristic, actual feasible completion)
        completion_val_greedy = _greedy_complete(new_cap, w_list, v_list, idx, mode='density')
        score_greedy = v_i + completion_val_greedy
        
        # Estimate 3: Greedy Value Completion
        completion_val_greedy_v = _greedy_complete(new_cap, w_list, v_list, idx, mode='value')
        score_greedy_v = v_i + completion_val_greedy_v
        
        # Combine scores.
        # The fractional bound is an upper bound, so it might overestimate.
        # The greedy completion is a lower bound on the best possible completion (since it's a specific feasible solution).
        # A good heuristic might balance these or pick the one that seems more "reliable".
        # Usually, the greedy completion is a better predictor of the *actual* final value because it accounts for the 0/1 constraint.
        # However, the fractional bound helps distinguish between items that leave "high density" capacity vs "low density" capacity.
        
        # Let's try a weighted combination or just use the greedy one if it's close to the bound.
        # Or simply maximize the greedy completion value, as that is the value we are *likely* to achieve.
        # But wait, the greedy completion depends on the order. If we pick item i, the remaining set is fixed.
        # The greedy completion on the remaining set is a specific value.
        # If we pick item j, the remaining set is different, and the greedy completion might be different.
        
        # Let's score based on the greedy completion value primarily, but penalize if the gap to the fractional bound is huge?
        # Or just use the fractional bound? The seed used fractional bound.
        # The inspirations used "dense candidate scoring" and "adaptive deep greedy estimation".
        
        # Let's try a composite score:
        # score = alpha * score_greedy + (1-alpha) * score_frac
        # Or maybe just score_greedy is better because it's a real solution value.
        # Let's test both. The seed (score 1.0) used fractional bound.
        # If fractional bound is too optimistic, it might pick items that look good in bound but lead to bad packing.
        # Greedy completion is more realistic.
        
        # Let's use a blend. If the fractional bound is much higher than the greedy, it suggests the remaining items don't pack well with the fractional assumption, so maybe trust the greedy more?
        # Actually, let's just use the greedy completion value as the primary score, as it represents a feasible completion.
        # But we should also consider that the greedy algorithm itself might not be optimal.
        
        # Let's try: score = score_greedy + 0.1 * score_frac
        # Or simply: score = score_greedy
        
        # Let's look at the inspirations again. "Bounded lookahead ... with dense candidate scoring".
        # This suggests evaluating candidates by how well they score under a dense (high density) heuristic.
        
        # Let's try scoring by the greedy completion value.
        current_score = score_greedy
        
        # Also consider the value/weight of the item itself? No, it's already in v_i.
        
        if current_score > best_score:
            best_score = current_score
            best_idx = idx
            
    return int(best_idx)
# EVOLVE-BLOCK-END
