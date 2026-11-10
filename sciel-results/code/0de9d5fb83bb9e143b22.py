"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 145.0  # safety margin before 150s kill

def _bnb_solve(capacity, weights, values):
    """
    Branch and Bound solver for 0/1 Knapsack.
    Returns a list of indices (in the original array) that form the optimal solution,
    or None if it times out or fails.
    """
    n = len(weights)
    if n == 0 or capacity <= 0:
        return []
    
    # Filter out items that don't fit individually? No, BnB handles that, 
    # but it's good for pruning.
    # Actually, let's just run BnB on all items.
    
    # Sort by value density descending
    densities = values / weights
    idx_sorted = np.argsort(-densities)
    sorted_w = weights[idx_sorted].astype(np.float64)
    sorted_v = values[idx_sorted].astype(np.float64)
    
    # Precompute suffix sums for fractional upper bound
    # suffix_w[i] = sum of weights from i to n-1
    # suffix_v[i] = sum of values from i to n-1
    suffix_w = np.zeros(n + 1)
    suffix_v = np.zeros(n + 1)
    if n > 0:
        suffix_w[n-1] = sorted_w[n-1]
        suffix_v[n-1] = sorted_v[n-1]
        for i in range(n-2, -1, -1):
            suffix_w[i] = suffix_w[i+1] + sorted_w[i]
            suffix_v[i] = suffix_v[i+1] + sorted_v[i]
            
    best_value = 0.0
    best_indices = []
    
    # Node: (level, current_value, current_weight, path_indices)
    # We use a stack for DFS, but BnB often benefits from priority queue (Best-First) 
    # or just DFS with good bounds. DFS is memory efficient.
    # Let's use a stack.
    stack = [(0, 0.0, 0.0, [])]
    
    # Time check helper
    time_check_interval = 1000
    check_count = 0
    
    while stack:
        check_count += 1
        if check_count % time_check_interval == 0:
            if time.time() - _START_TIME > _TIME_LIMIT - 5:
                return None # Timeout
        
        level, val, wt, path = stack.pop()
        
        if level >= n:
            if val > best_value:
                best_value = val
                best_indices = path.copy()
            continue
        
        # Upper bound calculation
        # Try to fill remaining capacity with fractional items starting from 'level'
        bound = val
        rem_cap = capacity - wt
        for i in range(level, n):
            if sorted_w[i] <= rem_cap:
                bound += sorted_v[i]
                rem_cap -= sorted_w[i]
            else:
                bound += sorted_v[i] * (rem_cap / sorted_w[i])
                break
        
        if bound <= best_value:
            continue
            
        # Branch 1: Include item 'level'
        if wt + sorted_w[level] <= capacity:
            new_val = val + sorted_v[level]
            new_wt = wt + sorted_w[level]
            new_path = path + [idx_sorted[level]]
            stack.append((level + 1, new_val, new_wt, new_path))
            
        # Branch 2: Exclude item 'level'
        stack.append((level + 1, val, wt, path))
        
    return best_indices

def _greedy_fallback(remaining_capacity, remaining_weights, remaining_values):
    """
    Fast greedy fallback if BnB times out or fails.
    Picks the item with highest density that fits, then simulates the rest to ensure validity.
    """
    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1
        
    feasible_mask = remaining_weights <= remaining_capacity
    if not np.any(feasible_mask):
        return -1
        
    feasible_idx = np.where(feasible_mask)[0]
    f_w = remaining_weights[feasible_mask]
    f_v = remaining_values[feasible_mask]
    
    densities = f_v / f_w
    best_idx_local = np.argmax(densities)
    return int(feasible_idx[best_idx_local])

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    """Pick the next item to add, or -1 to stop."""
    if remaining_capacity <= 0 or len(remaining_weights) == 0:
        return -1
    
    n = len(remaining_weights)
    
    # Check time budget
    if time.time() - _START_TIME > _TIME_LIMIT - 5:
        return -1
    
    # Feasible items only
    feasible_mask = remaining_weights <= remaining_capacity
    if not np.any(feasible_mask):
        return -1
        
    # Try to solve exactly with BnB
    # BnB is exponential in worst case, but very fast for small N or tight constraints.
    # With 100 items, it might be slow if capacity is large relative to weights.
    # However, capacity is 12.5, which is small.
    
    # Optimization: If the number of items is small, BnB is very fast.
    # If large, we might need to be careful.
    # Let's limit BnB to a certain number of items or just run it with a tight internal deadline.
    
    optimal_indices = _bnb_solve(remaining_capacity, remaining_weights, remaining_values)
    
    if optimal_indices is not None and len(optimal_indices) > 0:
        # Pick the first item from the optimal solution.
        # To be deterministic and potentially better for future steps, 
        # we can pick the one with the highest density among the optimal set, 
        # or just the first one.
        # Actually, any item from the optimal set is a valid "next" step 
        # that guarantees the global optimal total value is achievable.
        
        # Let's pick the item with the highest value density from the optimal set
        # to potentially leave more "flexible" capacity for the rest?
        # No, the remaining problem will be solved optimally by the next call.
        # So any item is fine. Let's just pick the first one returned.
        return int(optimal_indices[0])
        
    # Fallback to greedy if BnB timed out or returned empty (shouldn't happen if feasible)
    return _greedy_fallback(remaining_capacity, remaining_weights, remaining_values)
# EVOLVE-BLOCK-END
