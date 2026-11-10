import numpy as np
import time

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1
    
    start_time = time.time()
    DEADLINE = 120.0
    SAFETY_MARGIN = 5.0
    
    # Filter to feasible items only
    feasible = []
    for i in range(n):
        w = remaining_weights[i]
        v = remaining_values[i]
        if w > 0 and w <= remaining_capacity:
            feasible.append((v, w, i))
        elif w == 0 and v > 0:
            return i  # Free value, take it
    
    if not feasible:
        return -1
    
    # Build lookup dictionaries for O(1) access
    weight_lookup = {}
    value_lookup = {}
    for v, w, idx in feasible:
        weight_lookup[idx] = w
        value_lookup[idx] = v
    
    # Sort by density descending for greedy baseline
    feasible.sort(key=lambda x: x[0] / x[1], reverse=True)
    
    # Greedy fill to get a baseline
    greedy_val = 0
    greedy_items = []
    rem_cap = remaining_capacity
    for v, w, idx in feasible:
        if w <= rem_cap:
            greedy_val += v
            greedy_items.append(idx)
            rem_cap -= w
    
    best_first = feasible[0][2]
    best_total = greedy_val
    
    # Try each item as the first pick, then greedy fill the rest
    candidates = feasible[:min(50, len(feasible))]
    
    for v_first, w_first, idx_first in candidates:
        if w_first > remaining_capacity:
            continue
        rem_cap_after = remaining_capacity - w_first
        # Greedy fill with remaining items
        total = v_first
        for v, w, idx in feasible:
            if idx == idx_first:
                continue
            if w <= rem_cap_after:
                total += v
                rem_cap_after -= w
        if total > best_total:
            best_total = total
            best_first = idx_first
    
    # DP-based approach: find the optimal solution
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - SAFETY_MARGIN:
        # Scale weights to integers for DP
        SCALE = 100
        cap_scaled = int(round(remaining_capacity * SCALE))
        
        if 0 < cap_scaled < 10000:  # Only if DP is feasible
            n_feasible = len(feasible)
            
            # dp[w] = max value achievable with weight <= w
            dp = [0.0] * (cap_scaled + 1)
            # track[i][w] = 1 if item i was used to reach state w, 0 otherwise
            track = [[0] * (cap_scaled + 1) for _ in range(n_feasible)]
            
            # Process items in order
            for i, (v, w, idx) in enumerate(feasible):
                w_scaled = int(round(w * SCALE))
                if w_scaled > cap_scaled:
                    continue
                # Process in reverse to avoid using item twice
                for cw in range(cap_scaled, w_scaled - 1, -1):
                    if dp[cw - w_scaled] + v > dp[cw] + 1e-9:
                        dp[cw] = dp[cw - w_scaled] + v
                        track[i][cw] = 1
            
            # Reconstruct the optimal solution
            optimal_items = []
            cw = cap_scaled
            for i in range(n_feasible - 1, -1, -1):
                if track[i][cw] == 1:
                    idx = feasible[i][2]
                    optimal_items.append(idx)
                    w_scaled = int(round(feasible[i][1] * SCALE))
                    cw -= w_scaled
            
            dp_total = dp[cap_scaled]
            if dp_total > best_total + 1e-9 and optimal_items:
                best_total = dp_total
                # Return the item with highest density in the optimal solution
                best_density = -1
                for idx in optimal_items:
                    w = weight_lookup[idx]
                    v = value_lookup[idx]
                    density = v / w if w > 0 else float('inf')
                    if density > best_density:
                        best_density = density
                        best_first = idx
    
    # Local search: try swapping items
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - SAFETY_MARGIN:
        # Local search: try swapping items
        greedy_set = set(greedy_items)
        current_solution = list(greedy_items)
        current_value = greedy_val
        current_weight = sum(weight_lookup[idx] for idx in current_solution)
        
        improved = True
        max_swaps = 100
        swap_count = 0
        
        while improved and swap_count < max_swaps:
            improved = False
            swap_count += 1
            elapsed = time.time() - start_time
            if elapsed > DEADLINE - SAFETY_MARGIN:
                break
            
            # Try removing each item in the current solution
            for i, idx_out in enumerate(current_solution):
                if idx_out not in greedy_set:
                    continue
                w_out = weight_lookup[idx_out]
                v_out = value_lookup[idx_out]
                
                # Calculate new capacity after removing idx_out
                new_cap = remaining_capacity - (current_weight - w_out)
                
                # Try adding items not in the solution
                for v_in, w_in, idx_in in feasible:
                    if idx_in in greedy_set:
                        continue
                    if w_in <= new_cap:
                        new_value = current_value - v_out + v_in
                        if new_value > current_value + 1e-9:
                            # Perform the swap
                            current_solution[i] = idx_in
                            greedy_set.remove(idx_out)
                            greedy_set.add(idx_in)
                            current_value = new_value
                            current_weight = current_weight - w_out + w_in
                            improved = True
                            break
            
            if current_value > best_total:
                best_total = current_value
                # Find which item to return first - pick highest density
                best_density = -1
                for idx in current_solution:
                    w = weight_lookup[idx]
                    v = value_lookup[idx]
                    density = v / w
                    if density > best_density:
                        best_density = density
                        best_first = idx
    
    return best_first
