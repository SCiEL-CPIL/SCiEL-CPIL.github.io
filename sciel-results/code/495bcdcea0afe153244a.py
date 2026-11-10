"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import heapq

_START_TIME = time.time()
_TIME_LIMIT = 135.0  # Safety margin before 150s

def _fractional_bound(start_idx, rem_cap, sorted_w, sorted_v):
    """
    Compute the fractional knapsack upper bound starting from start_idx
    with remaining capacity rem_cap.
    sorted_w and sorted_v are arrays sorted by density descending.
    """
    if start_idx >= len(sorted_w) or rem_cap <= 0:
        return 0.0
    
    bound = 0.0
    for i in range(start_idx, len(sorted_w)):
        w = sorted_w[i]
        v = sorted_v[i]
        if w <= rem_cap:
            bound += v
            rem_cap -= w
        else:
            bound += v * (rem_cap / w)
            break
    return bound

def _bnb_solve(capacity, weights, values):
    """
    Branch and bound solver for 0/1 knapsack.
    Returns the optimal value and the list of original indices included.
    """
    n = len(weights)
    if n == 0 or capacity <= 0:
        return 0.0, []
    
    # Sort by density descending
    densities = values / weights
    sorted_indices = np.argsort(-densities)
    sorted_w = weights[sorted_indices].astype(np.float64)
    sorted_v = values[sorted_indices].astype(np.float64)
    orig_indices = sorted_indices.astype(np.int64)
    
    # Initial greedy solution
    greedy_val = 0.0
    greedy_included = []
    cap = capacity
    for i in range(n):
        if sorted_w[i] <= cap:
            greedy_val += sorted_v[i]
            cap -= sorted_w[i]
            greedy_included.append(i)
    
    best_value = greedy_val
    best_items = list(greedy_included)
    
    # Heap entries: (-bound, start_idx, current_value, current_weight, included_tuple)
    # We use tuples for included items to make them hashable/immutable for the heap
    heap = []
    root_bound = _fractional_bound(0, capacity, sorted_w, sorted_v)
    heapq.heappush(heap, (-root_bound, 0, 0.0, 0.0, ()))
    
    max_nodes = 2000  # Limit nodes to ensure speed per call
    
    while heap and max_nodes > 0:
        max_nodes -= 1
        neg_bound, idx, val, wt, included = heapq.heappop(heap)
        bound = -neg_bound
        
        if bound <= best_value:
            continue
        
        if idx >= n:
            if val > best_value:
                best_value = val
                best_items = list(included)
            continue
        
        # Try including item idx
        if sorted_w[idx] <= capacity - wt:
            new_wt = wt + sorted_w[idx]
            new_val = val + sorted_v[idx]
            new_included = included + (idx,)
            
            # Check if this is a better leaf (or intermediate)
            if new_val > best_value:
                best_value = new_val
                best_items = list(new_included)
            
            # Compute bound for child
            child_bound = _fractional_bound(idx + 1, capacity - new_wt, sorted_w, sorted_v) + new_val
            if child_bound > best_value:
                heapq.heappush(heap, (-child_bound, idx + 1, new_val, new_wt, new_included))
        
        # Try excluding item idx
        child_bound_excl = _fractional_bound(idx + 1, capacity - wt, sorted_w, sorted_v) + val
        if child_bound_excl > best_value:
            heapq.heappush(heap, (-child_bound_excl, idx + 1, val, wt, included))
    
    # Map back to original indices
    # best_items contains indices into the sorted array
    # We need to return the original indices in the order they were added? 
    # The problem asks for the "next item". The B&B gives the full set for the remaining subproblem.
    # We need to pick the FIRST item in the optimal solution for the remaining subproblem.
    # However, the "first" item is ambiguous. 
    # In the context of `select_next_item`, we are building a solution sequentially.
    # The B&B solves the subproblem optimally. 
    # We should return the item that is part of this optimal solution.
    # Which one? The one with the highest density? The one that appears first in the sorted order?
    # If we return the one with the highest density among the optimal set, it aligns with the greedy sort.
    # Actually, any item from the optimal set is a valid "next" item if the rest can be completed optimally.
    # But `select_next_item` only returns ONE index. The caller then updates state and calls again.
    # If we pick item A, the next call will solve the remaining problem optimally (hopefully).
    # So we should pick the item from the optimal set that is "best" in some sense, or just the first one found.
    # Let's pick the one with the highest density (lowest index in sorted order) to maintain consistency with the greedy sort.
    
    if not best_items:
        return 0.0, []
    
    # Find the item in best_items with the smallest index in the sorted array (highest density)
    min_sorted_idx = min(best_items)
    orig_idx = orig_indices[min_sorted_idx]
    
    return best_value, [int(orig_idx)]

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    # Deadline check
    if time.time() - _START_TIME > _TIME_LIMIT:
        # Fallback to simple greedy
        feasible = np.where(remaining_weights <= remaining_capacity)[0]
        if len(feasible) == 0:
            return -1
        densities = remaining_values[feasible] / remaining_weights[feasible]
        return int(feasible[np.argmax(densities)])

    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1

    # 1. Find feasible items
    feasible_mask = remaining_weights <= remaining_capacity
    feasible_indices = np.where(feasible_mask)[0]
    
    if len(feasible_indices) == 0:
        return -1
    
    # If only a few items, solve exactly with B&B
    # If many items, B&B might be too slow or the problem is large enough that heuristics are needed?
    # With n=100, B&B with a good bound is usually fast.
    # Let's try B&B on the feasible subset.
    
    # Extract feasible weights and values
    fw = remaining_weights[feasible_indices]
    fv = remaining_values[feasible_indices]
    
    # Map back to original indices
    # fw[i] corresponds to remaining_weights[feasible_indices[i]]
    
    # Run B&B
    # Note: B&B returns the optimal value and the original index of the "first" item to pick.
    # We need to be careful: the B&B solves the subproblem defined by fw, fv.
    # The returned index is into the fw/fv array? No, I implemented it to return original indices relative to the input arrays of _bnb_solve.
    # Let's adjust _bnb_solve to return the index into the input arrays of _bnb_solve.
    
    # Re-implementing _bnb_solve logic inline or adjusting return value.
    # Current _bnb_solve returns orig_indices[min_sorted_idx].
    # orig_indices is the mapping from sorted index to input index.
    # So it returns an index into the fw/fv arrays.
    
    val, idxs = _bnb_solve(remaining_capacity, fw, fv)
    
    if not idxs:
        # Fallback if B&B fails or finds nothing
        densities = fv / fw
        local_best = np.argmax(densities)
        return int(feasible_indices[local_best])
    
    local_best = idxs[0]
    return int(feasible_indices[local_best])

# EVOLVE-BLOCK-END
