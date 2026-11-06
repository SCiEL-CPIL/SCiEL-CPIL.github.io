import numpy as np
import time

# Global deadline guard
_start_time = time.time()
_TIME_LIMIT = 130.0  # Leave 20s safety margin from 150s limit

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    n = len(remaining_weights)
    if n == 0:
        return -1
    
    # Check deadline
    if time.time() - _start_time > _TIME_LIMIT:
        return -1
    
    # Convert to numpy arrays if not already
    if not isinstance(remaining_weights, np.ndarray):
        remaining_weights = np.array(remaining_weights, dtype=np.float64)
    if not isinstance(remaining_values, np.ndarray):
        remaining_values = np.array(remaining_values, dtype=np.float64)
    
    # Find feasible items
    feasible_mask = remaining_weights <= remaining_capacity
    if not np.any(feasible_mask):
        return -1
    
    feasible_indices = np.where(feasible_mask)[0]
    fw = remaining_weights[feasible_indices]
    fv = remaining_values[feasible_indices]
    
    if len(fw) == 0:
        return -1
    
    # Strategy: Use a bounded lookahead approach.
    # For each candidate item, estimate the total value if we pick it,
    # then greedily fill the remaining capacity with density-sorted items.
    # We limit the lookahead to a subset of top candidates by density
    # to keep the computation fast.
    
    # Compute densities
    densities = fv / fw
    
    # Sort by density descending
    order = np.argsort(densities)[::-1]
    sorted_fw = fw[order]
    sorted_fv = fv[order]
    sorted_densities = densities[order]
    
    # Pre-compute prefix sums of weights and values for the density-sorted list
    # This allows O(1) greedy fill estimation for a given capacity
    # when we can find the right cutoff via binary search.
    
    # For the greedy fill: given capacity C, we take items in density order
    # until we can't fit the next one. With prefix sums, we can binary search
    # for the largest k such that sum(sorted_fw[:k]) <= C.
    # Then value = sum(sorted_fv[:k]) + (C - sum(sorted_fw[:k])) * sorted_densities[k]
    # if k < len(sorted_fw), else just sum(sorted_fv[:k]).
    
    # But we need to exclude the candidate item from the fill.
    # So for each candidate, we need the greedy fill of the remaining items
    # excluding that candidate.
    
    # To make this efficient, we'll only consider the top K candidates by density
    # as potential picks. For each, we compute the greedy fill of the rest.
    
    # Let's set K to a reasonable limit based on time budget.
    # With n=100 items, even K=50 with O(n) fill each is 5000 ops, very fast.
    # But the fill itself needs to exclude one item, which complicates prefix sums.
    
    # Alternative: For each candidate i, the greedy fill of all items except i
    # can be computed by:
    # 1. Take all items in density order
    # 2. Skip item i
    # 3. Greedily fill with capacity remaining_capacity - fw[i]
    
    # This is O(n) per candidate, so O(K*n) total. With K=50, n=100, that's 5000 ops.
    
    # Let's implement this.
    
    K = min(50, len(order))
    
    best_score = -1
    best_index = -1
    
    for local_idx in range(K):
        # Candidate is order[local_idx]
        cand_global_in_feasible = order[local_idx]
        cand_w = fw[cand_global_in_feasible]
        cand_v = fv[cand_global_in_feasible]
        
        remaining_cap = remaining_capacity - cand_w
        if remaining_cap < 0:
            continue
        
        # Greedily fill with all feasible items except the candidate
        # in density order
        fill_value = 0.0
        current_cap = remaining_cap
        
        for j in range(len(order)):
            if j == local_idx:
                continue  # Skip the candidate
            item_w = sorted_fw[j]
            if item_w <= current_cap:
                fill_value += sorted_fv[j]
                current_cap -= item_w
            else:
                break
        
        total_value = cand_v + fill_value
        
        if total_value > best_score:
            best_score = total_value
            best_index = feasible_indices[cand_global_in_feasible]
    
    if best_index == -1:
        return -1
    
    # Sanity check
    if remaining_weights[best_index] > remaining_capacity:
        return -1
    
    return int(best_index)
