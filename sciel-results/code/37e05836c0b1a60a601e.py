"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Cache for DP solutions
# Key: (capacity, tuple(remaining_weights), tuple(remaining_values))
# Value: list of indices in the current remaining arrays to pick, in order
_dp_cache = {}
_start_time = time.time()
_TIME_LIMIT = 110.0  # Safety margin before 120s limit

def _solve_knapsack_dp(capacity, weights, values):
    """
    Solve 0/1 knapsack exactly using DP.
    Returns a list of indices (into the input arrays) to pick.
    """
    n = len(weights)
    if n == 0:
        return []
    
    # If capacity is very large or n is small, DP is feasible.
    # Weights and capacity are floats, but for DP we need discrete states.
    # However, the problem states capacity 12.5 and 100 items.
    # If weights are continuous, DP on weight is not directly applicable in the standard integer sense.
    # BUT, usually in these generated tasks, weights might be integers or simple fractions.
    # Let's check the nature of the inputs. The prompt says "capacity 12.5".
    # If weights are floats, we can scale them.
    # Alternatively, if n=100, 2^100 is too big.
    # Standard DP O(n * C) requires C to be an integer.
    
    # Let's try to detect if weights are integers or can be scaled to integers.
    # If not, we fall back to a strong heuristic or branch and bound.
    # Given the small capacity (12.5), if weights are >= 0.1, C_scaled <= 125.
    # If weights are >= 0.01, C_scaled <= 1250.
    # If weights are arbitrary floats, this is hard.
    
    # Let's assume the generated instances have rational weights or integers.
    # To be safe and fast, let's use a Branch and Bound with density ordering,
    # which is often exact for small n and C, or very close.
    
    # Actually, let's look at the constraints again.
    # "100-item instances with capacity 12.5".
    # If weights are random uniform [0, 1], many items won't fit.
    # If weights are random uniform [0, 0.1], many fit.
    
    # A robust approach for "exact" or near-exact in this small space:
    # Branch and Bound (B&B) with fractional knapsack upper bound.
    
    # Sort items by density
    indices = np.argsort(values / weights)[::-1]
    sorted_weights = weights[indices]
    sorted_values = values[indices]
    n = len(sorted_weights)
    
    # If n is small enough, we could do exhaustive search, but 100 is too big.
    # B&B is the way.
    
    best_value = 0
    best_indices = []
    
    # State: (node_id, current_index, current_weight, current_value, path)
    # To avoid recursion depth issues and overhead, we can use a stack.
    # But B&B with fractional bound is efficient.
    
    # Optimization: If the sum of all remaining values is <= best_value, prune.
    # Precompute suffix sums of values for the sorted array? 
    # The sorted array is fixed for this call.
    suffix_values = np.zeros(n + 1)
    suffix_values[n-1] = sorted_values[n-1]
    for i in range(n-2, -1, -1):
        suffix_values[i] = suffix_values[i+1] + sorted_values[i]
    
    # Stack elements: (index, curr_w, curr_v, path_list)
    # Start with index 0
    stack = [(0, 0.0, 0.0, [])]
    
    while stack:
        idx, curr_w, curr_v, path = stack.pop()
        
        # If we've considered all items
        if idx == n:
            if curr_v > best_value:
                best_value = curr_v
                best_indices = path
            continue
            
        # Prune if current value + max possible remaining value <= best_value
        # Max possible remaining value is bounded by fractional knapsack from idx onwards
        # But a simpler bound: sum of all remaining values
        if curr_v + suffix_values[idx] <= best_value:
            continue
            
        # Option 1: Exclude item idx
        # Push exclude branch
        stack.append((idx + 1, curr_w, curr_v, path))
        
        # Option 2: Include item idx if it fits
        if curr_w + sorted_weights[idx] <= capacity:
            new_w = curr_w + sorted_weights[idx]
            new_v = curr_v + sorted_values[idx]
            new_path = path + [indices[idx]]
            
            # Check if this path is better
            if new_v > best_value:
                best_value = new_v
                best_indices = new_path
                
            # Push include branch
            stack.append((idx + 1, new_w, new_v, new_path))
            
    # Convert best_indices (which are indices into the sorted array) back to original indices
    # Wait, `indices` maps sorted_pos -> original_idx.
    # So best_indices contains original indices.
    
    return best_indices

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    global _dp_cache, _start_time
    
    # Check time limit
    if time.time() - _start_time > _TIME_LIMIT:
        # Fallback to greedy density to ensure we return something valid quickly
        # Though if we hit time limit, we might be in trouble. 
        # The prompt says "return the best result found with a safety margin".
        # Since this function is called iteratively, we can't "return the best result" for the whole instance 
        # from inside a single call unless we cheat.
        # However, the evaluator calls this function repeatedly to build the solution.
        # If we run out of time, we should just pick a valid item or stop.
        # Returning -1 stops the process.
        return -1
        
    # Create a key for the cache
    # We need to be careful with floating point in keys.
    # Round weights and values for key stability?
    # If the same set of items remains, the solution is the same.
    w_key = tuple(np.round(remaining_weights, 6))
    v_key = tuple(np.round(remaining_values, 6))
    c_key = round(remaining_capacity, 6)
    
    key = (c_key, w_key, v_key)
    
    if key in _dp_cache:
        plan = _dp_cache[key]
        if not plan:
            return -1
        next_idx = plan[0]
        _dp_cache[key] = plan[1:]
        return int(next_idx)
    
    # Solve for the current state
    # Note: remaining_weights and remaining_values are numpy arrays or lists.
    # We pass them to the solver.
    solution_indices = _solve_knapsack_dp(remaining_capacity, remaining_weights, remaining_values)
    
    # Cache the full sequence
    _dp_cache[key] = solution_indices
    
    if not solution_indices:
        return -1
        
    return int(solution_indices[0])
# EVOLVE-BLOCK-END
