import sys
import time
import numpy as np

# Increase recursion limit just in case, though B&B depth is N=100
sys.setrecursionlimit(2000)

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    start_time = time.time()
    time_limit = 1.0  # 1 second per call is generous for 100 calls (100s total)
    
    weights = np.asarray(remaining_weights)
    values = np.asarray(remaining_values)
    n = len(weights)
    
    if n == 0 or remaining_capacity <= 0:
        return -1

    # Filter feasible items
    feasible_indices = np.where(weights <= remaining_capacity)[0]
    if len(feasible_indices) == 0:
        return -1
    
    # Prepare items: list of (weight, value, original_index)
    items = []
    for i in feasible_indices:
        items.append((weights[i], values[i], int(i)))
    
    # If only one item, return it
    if len(items) == 1:
        return items[0][2]

    # Multiple greedy ordering strategies for better lower bound
    # Strategy 1: density = value / weight
    # Strategy 2: weight-first (lightest first)
    # Strategy 3: value-first (highest value first)
    # Strategy 4: modified density = value / sqrt(weight)
    # Strategy 5: value / weight^2
    
    def greedy_fill(item_list, cap):
        """Try to fill capacity with items in given order, return (total_value, chosen_indices)"""
        curr_cap = cap
        curr_val = 0
        chosen = []
        for w, v, idx in item_list:
            if w <= curr_cap:
                curr_cap -= w
                curr_val += v
                chosen.append(idx)
        return curr_val, chosen
    
    # Generate multiple orderings
    orderings = []
    
    # Density order
    def density_key(item):
        w, v, idx = item
        if w == 0:
            return float('inf') if v > 0 else 0
        return v / w
    orderings.append(sorted(items, key=density_key, reverse=True))
    
    # Weight-first (lightest first)
    orderings.append(sorted(items, key=lambda x: x[0]))
    
    # Value-first (highest value first)
    orderings.append(sorted(items, key=lambda x: x[1], reverse=True))
    
    # Modified density: value / sqrt(weight)
    def mod_density_key(item):
        w, v, idx = item
        if w == 0:
            return float('inf') if v > 0 else 0
        return v / (w ** 0.5)
    orderings.append(sorted(items, key=mod_density_key, reverse=True))
    
    # Value / weight^2
    def density_sq_key(item):
        w, v, idx = item
        if w == 0:
            return float('inf') if v > 0 else 0
        return v / (w * w)
    orderings.append(sorted(items, key=density_sq_key, reverse=True))
    
    # Try all orderings and keep the best
    best_val = 0
    best_set = []
    for ordering in orderings:
        val, chosen = greedy_fill(ordering, remaining_capacity)
        if val > best_val:
            best_val = val
            best_set = chosen

    # Sort by density descending for B&B efficiency
    def get_density(item):
        w, v, idx = item
        if w == 0:
            return float('inf') if v > 0 else 0
        return v / w
        
    items.sort(key=get_density, reverse=True)
    
    # Branch and Bound
    n_items = len(items)
    
    # Helper to calculate fractional knapsack upper bound from index k with capacity cap
    def fractional_bound(k, cap):
        if k >= n_items:
            return 0
        total_val = 0
        # We are at index k in the sorted items list
        # We can take items k, k+1, ...
        # Greedily take whole items, then fraction of next
        for i in range(k, n_items):
            w, v, idx = items[i]
            if w <= cap:
                total_val += v
                cap -= w
            else:
                # Take fraction
                if w > 0:
                    total_val += v * (cap / w)
                cap = 0
                break
        return total_val

    # Recursive B&B
    # Using a non-local variable for time check is tricky, 
    # but we can check time in the helper or just rely on the outer loop if we iterated.
    # Since we must return from the function, we use a flag.
    
    timeout_occurred = [False]

    def solve(k, cap, val, chosen):
        nonlocal best_val, best_set, timeout_occurred
        
        if timeout_occurred[0]:
            return
            
        # Prune
        if val + fractional_bound(k, cap) <= best_val:
            return
            
        if k == n_items:
            if val > best_val:
                best_val = val
                best_set = chosen
            return

        # Check time
        if time.time() - start_time > time_limit:
            timeout_occurred[0] = True
            return

        w, v, idx = items[k]

        # Branch 1: Include item k
        if w <= cap:
            solve(k + 1, cap - w, val + v, chosen + [idx])
            
        # Branch 2: Exclude item k
        if not timeout_occurred[0]:
            solve(k + 1, cap, val, chosen)

    solve(0, remaining_capacity, 0, [])

    if best_set:
        # Return the index of the first item in the optimal set
        # We sort by original index to be deterministic
        return min(best_set)
    
    # Fallback if B&B failed or found nothing (shouldn't happen if greedy ran)
    return -1

# To prevent global state issues if this function is called in parallel (unlikely)
# or to clean up, we don't use globals.
