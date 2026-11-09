import sys
import time
import numpy as np

# Increase recursion limit just in case, though B&B depth is N=100
sys.setrecursionlimit(2000)

# Global deadline guard: The evaluator kills after 150s.
# We track the start time of the entire evaluation process to ensure we finish safely.
# Since select_next_item is called multiple times (up to 100 times), we need to be careful.
# However, the problem statement says "Each evaluation must finish within 120s".
# The previous code used a 1.0s limit per call. With 100 calls, that's 100s.
# To be safer and allow for potential overhead, we can use a global deadline.
# But since we don't know when the first call happens, we can just use a generous per-call limit
# or a global deadline if we assume the function is called sequentially.
# Let's stick to a per-call limit but make it slightly more aggressive or use a global time check.
# Actually, the previous code's 1.0s per call is quite safe. The issue might be that B&B is too slow
# or not finding good solutions quickly enough.

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    start_time = time.time()
    # Use a tighter time limit to ensure we don't exceed the global budget if called many times.
    # 100 items * 1.0s = 100s. 150s limit. 50s buffer.
    # Let's try 0.5s per call to be safer? Or keep 1.0s but optimize B&B.
    time_limit = 0.5
    
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

    # Sort by density descending for B&B efficiency
    # density = value / weight
    # Handle weight 0 (shouldn't happen, but just in case)
    def get_density(item):
        w, v, idx = item
        if w == 0:
            return float('inf') if v > 0 else 0
        return v / w
        
    items.sort(key=get_density, reverse=True)
    
    # Greedy initialization for lower bound
    best_val = 0
    best_set = []
    curr_cap = remaining_capacity
    curr_val = 0
    greedy_set = []
    for w, v, idx in items:
        if w <= curr_cap:
            curr_cap -= w
            curr_val += v
            greedy_set.append(idx)
        else:
            break
    if curr_val > best_val:
        best_val = curr_val
        best_set = greedy_set

    # Local Search Polish (1-swap) to improve initial LB
    # This helps prune the B&B tree more effectively
    improved = True
    # We need to track which items are in the set for quick lookup
    # Since N is small, a set of indices is fine.
    current_set = set(best_set)
    current_val = best_val
    current_cap = remaining_capacity
    
    # Calculate current weight sum to check capacity for swaps easily?
    # No, we have current_cap.
    # For a swap i (out) -> j (in):
    # New cap = current_cap - w_j + w_i
    # Condition: current_cap - w_j + w_i >= 0
    # Value change: v_j - v_i > 0
    
    # We need to iterate through items not in set (candidates to add)
    # and items in set (candidates to remove)
    # To make this efficient, we can just loop through all pairs.
    # N <= 100, so N^2 = 10000 checks. Very fast.
    
    while improved:
        improved = False
        # Create lists for fast access
        in_set = [item for item in items if item[2] in current_set]
        out_set = [item for item in items if item[2] not in current_set]
        
        for out_item in in_set:
            w_out, v_out, idx_out = out_item
            for in_item in out_set:
                w_in, v_in, idx_in = in_item
                # Check if swap is feasible and improves value
                if v_in > v_out and (current_cap - w_in + w_out) >= 0:
                    # Perform swap
                    current_set.remove(idx_out)
                    current_set.add(idx_in)
                    current_val += v_in - v_out
                    current_cap -= w_in - w_out
                    improved = True
                    # Restart loop to check for further improvements
                    break
            if improved:
                break
        
        if time.time() - start_time > time_limit * 0.1: # Small time check for polish
            break

    if current_val > best_val:
        best_val = current_val
        best_set = list(current_set)

    # Branch and Bound
    n_items = len(items)
    
    # Precompute suffix sums for faster fractional bound calculation?
    # fractional_bound(k, cap) sums values from k to n_items-1 while subtracting weights.
    # We can precompute suffix_value[i] = sum of values from i to end
    # and suffix_weight[i] = sum of weights from i to end
    # But the fractional bound depends on cap, so we still need to iterate to find where cap runs out.
    # However, we can use the suffix sums to quickly check if all remaining items fit.
    
    suffix_value = np.zeros(n_items + 1)
    suffix_weight = np.zeros(n_items + 1)
    for i in range(n_items - 1, -1, -1):
        suffix_value[i] = suffix_value[i+1] + items[i][1]
        suffix_weight[i] = suffix_weight[i+1] + items[i][0]

    def fractional_bound(k, cap):
        if k >= n_items:
            return 0
        
        # If all remaining items fit, return suffix_value
        if suffix_weight[k] <= cap:
            return suffix_value[k]
            
        # Otherwise, iterate to find the fractional item
        total_val = 0
        for i in range(k, n_items):
            w, v, idx = items[i]
            if w <= cap:
                total_val += v
                cap -= w
            else:
                if w > 0:
                    total_val += v * (cap / w)
                cap = 0
                break
        return total_val

    # Recursive B&B with iterative deepening or just standard recursion?
    # Standard recursion can hit depth 100, which is fine.
    
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
