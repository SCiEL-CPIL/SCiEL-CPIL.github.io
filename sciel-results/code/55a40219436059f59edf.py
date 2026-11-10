import numpy as np
import time
import heapq

_GLOBAL_DEADLINE = None
_GLOBAL_ELAPSED = 0.0

def _check_deadline():
    global _GLOBAL_DEADLINE
    if _GLOBAL_DEADLINE is None:
        return True
    return (time.time() - _GLOBAL_ELAPSED) < (_GLOBAL_DEADLINE - 5.0)  # 5s safety margin

def _solve_knapsack_bnb(capacity, weights, values):
    """
    Branch and bound for 0/1 knapsack.
    Returns (optimal_value, list_of_indices_included).
    """
    n = len(weights)
    if n == 0 or capacity <= 0:
        return 0.0, []
    
    # Sort by value density descending
    idx_sorted = np.argsort(-values / weights)
    sorted_w = weights[idx_sorted].astype(np.float64)
    sorted_v = values[idx_sorted].astype(np.float64)
    
    # Precompute prefix sums for fast fractional bound
    prefix_w = np.zeros(n + 1)
    prefix_v = np.zeros(n + 1)
    for i in range(n):
        prefix_w[i+1] = prefix_w[i] + sorted_w[i]
        prefix_v[i+1] = prefix_v[i] + sorted_v[i]
    
    # Fractional bound using binary search:
    # Find the last item we can fully take, then fractionally take the next
    def fractional_bound(start_idx, rem_cap):
        if start_idx >= n or rem_cap <= 0:
            return 0.0
        
        # We want to find the largest j >= start_idx such that sum(w[start_idx..j-1]) <= rem_cap
        # i.e., prefix_w[j] - prefix_w[start_idx] <= rem_cap
        # i.e., prefix_w[j] <= prefix_w[start_idx] + rem_cap
        target = prefix_w[start_idx] + rem_cap
        
        # Binary search for the rightmost j where prefix_w[j] <= target
        # prefix_w is non-decreasing
        lo = start_idx
        hi = n
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if prefix_w[mid] <= target:
                lo = mid
            else:
                hi = mid - 1
        
        # lo is the largest index where we can fully take items start_idx..lo-1
        # Items fully taken: start_idx to lo-1
        val_full = prefix_v[lo] - prefix_v[start_idx]
        
        # If lo < n, we can fractionally take item lo
        if lo < n:
            remaining_after_full = rem_cap - (prefix_w[lo] - prefix_w[start_idx])
            if remaining_after_full > 0 and sorted_w[lo] > 0:
                val_full += sorted_v[lo] * (remaining_after_full / sorted_w[lo])
        
        return val_full
    
    # Initial greedy solution
    greedy_val = 0.0
    greedy_indices = []
    cap = capacity
    for i in range(n):
        if sorted_w[i] <= cap:
            greedy_val += sorted_v[i]
            cap -= sorted_w[i]
            greedy_indices.append(i)
    
    # B&B with max-heap (use negative for min-heap)
    # Each entry: (-bound, start_idx, current_value, current_weight, included_set)
    best_value = greedy_val
    best_items = list(greedy_indices)
    
    # Start with root node
    heap = []
    root_bound = fractional_bound(0, capacity)
    heapq.heappush(heap, (-root_bound, 0, 0.0, 0.0, frozenset()))
    
    max_nodes = 50000  # Increased node limit; deadline guard prevents timeout
    
    while heap and max_nodes > 0:
        if not _check_deadline():
            break
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
        
        # Branch 1: Include item idx
        if sorted_w[idx] <= capacity - wt:
            new_wt = wt + sorted_w[idx]
            new_val = val + sorted_v[idx]
            new_included = included | {idx}
            next_idx = idx + 1
            if next_idx >= n:
                if new_val > best_value:
                    best_value = new_val
                    best_items = list(new_included)
            else:
                bound_next = new_val + fractional_bound(next_idx, capacity - new_wt)
                if bound_next > best_value:
                    heapq.heappush(heap, (-bound_next, next_idx, new_val, new_wt, new_included))
        
        # Branch 2: Exclude item idx
        next_idx = idx + 1
        if next_idx >= n:
            if val > best_value:
                best_value = val
                best_items = list(included)
        else:
            bound_next = val + fractional_bound(next_idx, capacity - wt)
            if bound_next > best_value:
                heapq.heappush(heap, (-bound_next, next_idx, val, wt, included))
    
    # Map back to original indices
    result_indices = [int(idx_sorted[i]) for i in best_items]
    return best_value, result_indices


def _greedy_density(capacity, weights, values):
    """Simple density greedy."""
    n = len(weights)
    if n == 0 or capacity <= 0:
        return 0.0, []
    
    idx_sorted = np.argsort(-values / weights)
    total = 0.0
    cap = capacity
    result = []
    for i in idx_sorted:
        if weights[i] <= cap:
            total += values[i]
            cap -= weights[i]
            result.append(int(i))
    return total, result


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    global _GLOBAL_DEADLINE, _GLOBAL_ELAPSED
    
    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1
    
    # Find feasible items
    feasible_mask = remaining_weights <= remaining_capacity
    feasible_idx = np.where(feasible_mask)[0]
    
    if len(feasible_idx) == 0:
        return -1
    
    # Set up global deadline on first call
    if _GLOBAL_DEADLINE is None:
        _GLOBAL_ELAPSED = time.time()
        _GLOBAL_DEADLINE = _GLOBAL_ELAPSED + 140.0  # 140s out of 150s budget
    
    # Try B&B with global deadline check
    try:
        opt_val, opt_indices = _solve_knapsack_bnb(remaining_capacity, remaining_weights, remaining_values)
    except Exception:
        opt_val, opt_indices = _greedy_density(remaining_capacity, remaining_weights, remaining_values)
    
    if len(opt_indices) == 0:
        # Fall back to density greedy
        _, greedy_indices = _greedy_density(remaining_capacity, remaining_weights, remaining_values)
        if len(greedy_indices) == 0:
            return -1
        return greedy_indices[0]
    
    # Pick the item from the optimal solution that maximizes value density
    # among those in the optimal set — this helps maintain a consistent
    # greedy-like trajectory while respecting the global optimum
    best_density = -1
    best_choice = opt_indices[0]
    for idx in opt_indices:
        dens = remaining_values[idx] / remaining_weights[idx]
        if dens > best_density:
            best_density = dens
            best_choice = idx
    
    return int(best_choice)
