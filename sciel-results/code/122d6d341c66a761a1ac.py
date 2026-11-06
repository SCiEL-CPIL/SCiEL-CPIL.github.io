"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 145.0  # 5s margin before 150s limit

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    """
    Heuristic for 0/1 knapsack using value-scaled dynamic programming.
    """
    if time.time() > _DEADLINE:
        return -1
    
    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1
    
    # Filter to feasible items
    feasible_mask = remaining_weights <= remaining_capacity
    if not np.any(feasible_mask):
        return -1
    
    feasible_indices = np.where(feasible_mask)[0]
    feasible_weights = remaining_weights[feasible_mask]
    feasible_values = remaining_values[feasible_mask]
    
    if len(feasible_indices) == 0:
        return -1
        
    # If few items, we can use higher resolution or exact search if possible
    # Determine scaling factor S. Smaller S -> better accuracy, higher cost.
    # Total scaled value approx sum(values) / S
    # We want sum(values)/S to be around 500-2000 for speed/accuracy balance.
    total_val = np.sum(feasible_values)
    if total_val == 0:
        return -1
        
    # Target DP size
    target_dp_size = 1000
    S = max(1.0, total_val / target_dp_size)
    
    # Scale values to integers
    scaled_values = np.floor(feasible_values / S).astype(int)
    
    max_scaled_val = int(np.max(scaled_values)) if len(scaled_values) > 0 else 0
    if max_scaled_val == 0:
        # All values are 0 after scaling? Just pick highest value item
        max_val_idx = np.argmax(feasible_values)
        return int(feasible_indices[max_val_idx])

    # DP: dp[v] = minimum weight to achieve scaled value v
    # Size: sum of scaled values + 1
    sum_scaled = int(np.sum(scaled_values))
    if sum_scaled > 5000: # Safety cap
         S = total_val / 5000
         scaled_values = np.floor(feasible_values / S).astype(int)
         sum_scaled = int(np.sum(scaled_values))
         
    dp_size = sum_scaled + 1
    # Initialize with infinity (large weight)
    dp = np.full(dp_size, remaining_capacity + 1.0, dtype=np.float64)
    dp[0] = 0.0
    
    # Store backtracking info: parent pointer for each value level?
    # Since it's 0/1 knapsack, we need to know which items were taken.
    # Standard backtracking requires storing the state or doing a second pass.
    # Given small n (<=100) and small dp_size, we can store a bitset or list of items.
    # But storing lists is expensive.
    # Alternative: Just find the optimal value, then greedily check which items are "essential"?
    # Actually, we just need to pick ONE item that is part of SOME optimal solution.
    # We can do this:
    # 1. Find max achievable scaled value V_opt such that dp[V_opt] <= remaining_capacity.
    # 2. For each item i, check if there exists a solution of value V_opt that includes item i.
    #    This is equivalent to: dp_before_item_i[V_opt - scaled_val_i] <= remaining_capacity - weight_i
    #    where dp_before_item_i is the DP state before processing item i.
    #    We don't have dp_before_item_i stored.
    
    # Simpler approach:
    # Run the DP. Get V_opt.
    # Then, for each item i, simulate removing it and see if we can still achieve V_opt with remaining items?
    # That's O(n^2 * DP) which is too slow.
    
    # Better: During DP, we can track the last item added for each value?
    # No, 0/1 knapsack backtracking is tricky without storing the full table history.
    
    # Let's use a different backtracking method:
    # We compute the DP table.
    # Then we iterate through items in reverse order and determine if they were included.
    # To do this, we need the DP table at each step. Storing n x dp_size is 100 * 1000 = 100k floats. ~800KB. This is fine.
    
    # Re-run DP storing history
    history = np.zeros((n, dp_size), dtype=np.float64)
    dp = np.full(dp_size, remaining_capacity + 1.0, dtype=np.float64)
    dp[0] = 0.0
    history[0] = dp.copy()
    
    for i in range(n):
        w = feasible_weights[i]
        v = scaled_values[i]
        if v == 0:
            history[i+1] = dp.copy()
            continue
        # Update dp in reverse to avoid using item twice
        # dp_new[v] = min(dp_old[v], dp_old[v-v_i] + w_i)
        # We update dp in place, but we need the previous state for backtracking.
        # So we copy dp to history[i+1] AFTER update? No, history[i] is state before item i.
        # Let's store state BEFORE each item.
        # history[0] is initial.
        # history[i+1] is state after processing item i.
        
        # We need to update dp.
        # To do in-place reverse update:
        for val in range(dp_size - 1, v - 1, -1):
            prev_val = dp[val - v]
            if prev_val + w < dp[val]:
                dp[val] = prev_val + w
        history[i+1] = dp.copy()
        
    # Find optimal scaled value
    # We want max v such that dp[v] <= remaining_capacity
    valid_values = np.where(dp <= remaining_capacity)[0]
    if len(valid_values) == 0:
         # Fallback to greedy
         densities = feasible_values / feasible_weights
         max_idx = np.argmax(densities)
         return int(feasible_indices[max_idx])
         
    V_opt = int(valid_values[-1])
    
    # Backtrack to find which items are in the optimal solution
    # We check if item i was used.
    # Item i was used if history[i+1][V_opt] == history[i][V_opt - scaled_values[i]] + feasible_weights[i]
    # And history[i][V_opt - scaled_values[i]] <= remaining_capacity (which is true if the sum is)
    
    optimal_items = []
    current_v = V_opt
    for i in range(n - 1, -1, -1):
        v_i = scaled_values[i]
        w_i = feasible_weights[i]
        
        # Check if item i is part of the solution
        # The value at history[i+1][current_v] should be achievable by taking item i
        # from history[i][current_v - v_i]
        
        if v_i > current_v:
            continue
            
        prev_v = current_v - v_i
        prev_w = history[i][prev_v]
        
        # If the min weight to achieve current_v after item i is equal to 
        # min weight to achieve prev_v before item i + weight of item i
        # Then item i is a candidate for being in the optimal path.
        # Note: There might be multiple optimal paths. We just need one.
        
        if abs(history[i+1][current_v] - (prev_w + w_i)) < 1e-9:
            optimal_items.append(i)
            current_v = prev_v
            
        if current_v == 0:
            break
            
    if not optimal_items:
        # Fallback: pick item with highest density
        densities = feasible_values / feasible_weights
        max_idx = np.argmax(densities)
        return int(feasible_indices[max_idx])
        
    # We have a set of items that form an optimal solution (one of them).
    # We need to pick the NEXT item to add.
    # Any item in this set is a valid choice that leads to an optimal total value.
    # To be safe and consistent, pick the one with the highest value density from this set.
    
    best_idx_in_set = -1
    best_density = -1.0
    
    for idx in optimal_items:
        w = feasible_weights[idx]
        v = feasible_values[idx]
        density = v / w if w > 0 else float('inf')
        if density > best_density:
            best_density = density
            best_idx_in_set = idx
            
    return int(feasible_indices[best_idx_in_set])

# EVOLVE-BLOCK-END
