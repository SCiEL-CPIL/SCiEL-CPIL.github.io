import numpy as np
import time

_START = None
_HARD_LIMIT = 145.0  # Safety margin before 150s kill

def _now():
    global _START
    if _START is None:
        _START = time.time()
    return time.time()

def _fractional_bound(weights, values, capacity):
    """
    Computes the upper bound of the knapsack problem assuming items can be
    fractionally selected. This is used as a surrogate for the optimal value
    of the remaining items given a certain capacity.
    """
    n = len(weights)
    if n == 0 or capacity <= 0:
        return 0.0
    
    # Filter out items that are too heavy to fit even fractionally? 
    # Actually, fractional knapsack takes any part. But if w > cap, we can still take cap/w fraction.
    # However, standard fractional knapsack sorts by density.
    
    # We need to be careful with zero weights
    valid_mask = weights > 0
    if not np.all(valid_mask):
        # Handle zero weight items: they are always taken if value > 0
        zero_w_val = np.sum(values[~valid_mask])
        w_pos = weights[valid_mask]
        v_pos = values[valid_mask]
    else:
        zero_w_val = 0.0
        w_pos = weights
        v_pos = values
        
    if len(w_pos) == 0:
        return zero_w_val
        
    # Calculate densities
    with np.errstate(divide='ignore', invalid='ignore'):
        densities = v_pos / w_pos
    
    # Sort by density descending
    order = np.argsort(-densities, kind='stable')
    
    total_val = zero_w_val
    remaining_cap = capacity
    
    for idx in order:
        if remaining_cap <= 1e-9:
            break
        wi = w_pos[idx]
        vi = v_pos[idx]
        
        if wi <= remaining_cap:
            total_val += vi
            remaining_cap -= wi
        else:
            # Take fraction
            total_val += vi * (remaining_cap / wi)
            remaining_cap = 0
            break
            
    return total_val

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    if remaining_capacity is None:
        return -1
    
    cap = float(remaining_capacity)
    weights = np.asarray(remaining_weights, dtype=float)
    values = np.asarray(remaining_values, dtype=float)
    n = len(weights)
    
    if n == 0 or cap <= 0:
        return -1
        
    # Check if any item is feasible
    feasible_mask = weights <= cap
    if not np.any(feasible_mask):
        return -1
        
    feasible_indices = np.where(feasible_mask)[0]
    
    # If time is running out, fallback to simple density greedy
    if _now() - _START > _HARD_LIMIT - 5.0:
        # Simple density pick
        with np.errstate(divide='ignore', invalid='ignore'):
            density = np.where(weights > 0, values / weights, 0.0)
        # Only consider feasible ones
        density_feasible = np.where(feasible_mask, density, -np.inf)
        return int(np.argmax(density_feasible))

    # Lagrangian Relaxation Strategy:
    # For each candidate item i, compute:
    # Value(i) + FractionalBound(remaining_items \ {i}, cap - w[i])
    # Select i that maximizes this score.
    
    # To speed this up, we can't do a full O(N^2 log N) loop for N=100 if called 100 times (10^6 * log operations).
    # 100 * 100 * log(100) is roughly 100 * 100 * 7 = 70,000 operations per step.
    # 100 steps * 70,000 = 7,000,000 operations. This is very fast in Python/Numpy.
    
    best_score = -np.inf
    best_idx = -1
    
    # We can optimize by pre-sorting or just doing the loop.
    # Let's do the loop over feasible candidates.
    
    # To avoid modifying arrays in place repeatedly which is slow, 
    # we can compute the bound on the full set minus one item.
    # However, fractional bound is sensitive to which items are present.
    
    # Optimization: 
    # The fractional bound of the remaining set after removing item i 
    # is very similar to the fractional bound of the full set.
    # A cheaper approximation: 
    # Score(i) = v[i] + Bound(full_set, cap - w[i])
    # But this double counts or misestimates because item i is still in the "full_set" for the bound calculation 
    # if we don't remove it. 
    # Actually, if we remove i, the bound might drop slightly.
    # For a heuristic, computing the bound on the *remaining* items (excluding i) is more accurate.
    
    # Let's try the accurate but potentially slower O(N^2 log N) per step approach first, 
    # as N=100 is small.
    
    # Pre-compute some things? 
    # Actually, just iterating over feasible candidates and computing the bound for the rest is fine.
    
    # We can speed up the bound calculation by using a pre-sorted order of ALL items by density.
    # Then for a specific subset (all except i), we just iterate through the global order 
    # and skip item i.
    
    # 1. Global sort by density
    with np.errstate(divide='ignore', invalid='ignore'):
        global_densities = np.where(weights > 0, values / weights, 0.0)
    
    # Items with 0 weight and positive value should always be picked first.
    # But since we are picking ONE item, and 0-weight items don't reduce capacity,
    # picking one doesn't hurt. Picking a heavy item might hurt.
    # So, if there are 0-weight items, we should probably pick one with highest value?
    # Or does it matter? 
    # If we pick a 0-weight item, cap remains same. 
    # If we pick a heavy item, cap decreases.
    # So 0-weight items are "free". We should prioritize them if they have high value?
    # Actually, the Lagrangian score for a 0-weight item i:
    # v[i] + Bound(rest, cap)
    # For a heavy item j:
    # v[j] + Bound(rest, cap - w[j])
    # Since Bound(rest, cap) >= Bound(rest, cap - w[j]), the 0-weight item has a structural advantage 
    # unless v[j] is huge.
    
    # Let's implement the efficient bound calculation using global order.
    
    # Sort indices by density descending
    order = np.argsort(-global_densities, kind='stable')
    
    # Pre-compute prefix sums? No, because we skip different items each time.
    # But we can just iterate through `order` and skip the candidate.
    
    # For each feasible candidate i:
    for i in feasible_indices:
        wi = weights[i]
        vi = values[i]
        
        # Calculate fractional bound for items \ {i} with capacity cap - wi
        rem_cap = cap - wi
        
        bound_val = 0.0
        current_cap = rem_cap
        
        # Iterate through global density order
        # We need to skip index i
        for idx in order:
            if idx == i:
                continue
            if current_cap <= 1e-9:
                break
            
            w_idx = weights[idx]
            v_idx = values[idx]
            
            # Note: In the bound calculation, we consider ALL remaining items, 
            # even if they were originally infeasible in the 0/1 sense? 
            # Fractional knapsack allows taking parts. 
            # But if w_idx > current_cap, we take a fraction.
            # This is correct for the LP relaxation.
            
            if w_idx <= current_cap:
                bound_val += v_idx
                current_cap -= w_idx
            else:
                if w_idx > 0:
                    bound_val += v_idx * (current_cap / w_idx)
                    current_cap = 0
                # If w_idx == 0, we take all value
                else:
                    bound_val += v_idx
                    # current_cap unchanged
            
        total_score = vi + bound_val
        
        if total_score > best_score:
            best_score = total_score
            best_idx = i
            
        # Time check inside loop? 
        # N=100, inner loop N=100. 10,000 ops per candidate. 100 candidates. 1,000,000 ops per step.
        # 100 steps -> 100,000,000 ops. Might be too slow in Python?
        # 100 million simple float ops in Python can take ~5-10 seconds. 
        # 100 items * 100 steps = 100 calls to select_next_item.
        # Wait, select_next_item is called ONCE per item added.
        # So it's called ~50-80 times (since we stop when cap is full or no items fit).
        # 80 calls * 1,000,000 ops = 80,000,000 ops.
        # This should fit within 150s easily.
        
        if _now() - _START > _HARD_LIMIT - 2.0:
            # Fallback if we are running out of time
            # Just pick the best one found so far, or break and use density
            break

    if best_idx == -1:
        # Fallback
        with np.errstate(divide='ignore', invalid='ignore'):
            density = np.where(weights > 0, values / weights, 0.0)
        density_feasible = np.where(feasible_mask, density, -np.inf)
        best_idx = int(np.argmax(density_feasible))
        
    return int(best_idx)
