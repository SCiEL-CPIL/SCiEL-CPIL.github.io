import numpy as np
import time

_START_TIME = time.time()
_DEADLINE_SECONDS = 120.0
_SAFETY_MARGIN = 5.0

def _elapsed_ok():
    return (time.time() - _START_TIME) < (_DEADLINE_SECONDS - _SAFETY_MARGIN)

def _greedy_fill_value(capacity, weights, values, sorted_indices, exclude_set):
    """
    Greedily fill the capacity with items from sorted_indices,
    skipping any index in exclude_set.
    Returns the total value.
    """
    total_val = 0.0
    rem_cap = capacity
    for idx in sorted_indices:
        if idx in exclude_set:
            continue
        w = weights[idx]
        if w <= rem_cap:
            rem_cap -= w
            total_val += values[idx]
            if rem_cap <= 0:
                break
    return total_val

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1
    
    if not _elapsed_ok():
        # Fallback: simple density greedy
        densities = remaining_values / remaining_weights
        feasible = remaining_weights <= remaining_capacity
        if not np.any(feasible):
            return -1
        masked_densities = np.where(feasible, densities, -1)
        return int(np.argmax(masked_densities))

    # Pre-compute densities and sort indices
    densities = remaining_values / remaining_weights
    sorted_indices = np.argsort(-densities)
    
    best_total_value = -1.0
    best_index = -1
    
    # Define a set of top candidates for deeper lookahead
    # We will do 2-step lookahead for the top 20 density items
    TOP_K = 20
    top_k_indices = set(sorted_indices[:min(TOP_K, n)])
    
    for i in range(n):
        w_i = remaining_weights[i]
        if w_i > remaining_capacity:
            continue
        
        v_i = remaining_values[i]
        rem_cap_after_i = remaining_capacity - w_i
        
        if i in top_k_indices:
            # 2-step lookahead
            best_sub_val = 0.0
            # Try picking a second item j
            for j in sorted_indices:
                if j == i:
                    continue
                w_j = remaining_weights[j]
                if w_j > rem_cap_after_i:
                    continue
                
                # If we pick i and j, remaining capacity is rem_cap_after_i - w_j
                rem_cap_after_j = rem_cap_after_i - w_j
                v_j = remaining_values[j]
                
                # Fill the rest greedily, excluding i and j
                # To optimize, we can break early if rem_cap_after_j is small
                fill_val = _greedy_fill_value(rem_cap_after_j, remaining_weights, remaining_values, sorted_indices, {i, j})
                
                total_val = v_i + v_j + fill_val
                if total_val > best_sub_val:
                    best_sub_val = total_val
            
            # If no j was picked (all too heavy), best_sub_val is just v_i + greedy_fill(rem_cap_after_i, exclude {i})
            # Wait, the loop above handles the case where j is picked.
            # We also need to consider the case where we pick i and then just greedy fill without a specific "second step" choice?
            # The 2-step lookahead effectively chooses the best j. If no j fits, the loop doesn't update best_sub_val from 0?
            # No, best_sub_val starts at 0. If no j fits, we should just do 1-step greedy.
            # Let's handle the 1-step case as a baseline for the "pick i" decision.
            
            # 1-step value for i
            val_1step = v_i + _greedy_fill_value(rem_cap_after_i, remaining_weights, remaining_values, sorted_indices, {i})
            
            # The 2-step value is the max of (v_i + v_j + fill) for all valid j.
            # If no j is valid, best_sub_val remains 0, which is wrong.
            # We should initialize best_sub_val to val_1step.
            # Actually, if we pick i, the best we can do is at least val_1step.
            # The loop finds if there's a better specific j.
            
            if best_sub_val < val_1step:
                best_sub_val = val_1step
                
            current_total = best_sub_val
        else:
            # 1-step lookahead for others
            current_total = v_i + _greedy_fill_value(rem_cap_after_i, remaining_weights, remaining_values, sorted_indices, {i})
            
        if current_total > best_total_value:
            best_total_value = current_total
            best_index = i
            
    return best_index
