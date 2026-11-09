"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global start time for safety
_start_time = time.time()
_TIMEOUT = 145.0  # Safety margin before 150s kill

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1

    # Safety check: if we've been running too long, fall back to simple greedy
    if time.time() - _start_time > _TIMEOUT:
        weights = np.array(remaining_weights, dtype=float)
        values = np.array(remaining_values, dtype=float)
        with np.errstate(divide='ignore', invalid='ignore'):
            densities = values / weights
        order = np.argsort(-densities)
        for idx in order:
            if weights[idx] <= remaining_capacity:
                return int(idx)
        return -1

    weights = np.array(remaining_weights, dtype=float)
    values = np.array(remaining_values, dtype=float)
    
    # Pre-compute sorting orders
    # 1. Density (Value / Weight)
    with np.errstate(divide='ignore', invalid='ignore'):
        densities = values / weights
    order_density = np.argsort(-densities)
    
    # 2. Value (Highest Value first)
    order_value = np.argsort(-values)
    
    # 3. Weight (Lightest first)
    order_weight = np.argsort(weights)
    
    best_score = -1.0
    best_idx = -1
    
    for i in range(n):
        # Check if item i is feasible
        if weights[i] > remaining_capacity:
            continue
            
        cap = remaining_capacity - weights[i]
        base_val = values[i]
        
        max_sim_val = 0.0
        
        # Simulate 1: Fractional Density Greedy
        # Fractional knapsack provides a tighter upper bound for the remaining capacity
        sim_cap = cap
        sim_val = 0.0
        for j in order_density:
            if j == i:
                continue
            if weights[j] <= sim_cap:
                sim_cap -= weights[j]
                sim_val += values[j]
            else:
                # Take fraction of the item that fills the remaining capacity
                if weights[j] > 0:
                    sim_val += values[j] * (sim_cap / weights[j])
                break
        if sim_val > max_sim_val:
            max_sim_val = sim_val
            
        # Simulate 2: Fractional Value Greedy
        sim_cap = cap
        sim_val = 0.0
        for j in order_value:
            if j == i:
                continue
            if weights[j] <= sim_cap:
                sim_cap -= weights[j]
                sim_val += values[j]
            else:
                if weights[j] > 0:
                    sim_val += values[j] * (sim_cap / weights[j])
                break
        if sim_val > max_sim_val:
            max_sim_val = sim_val

        # Simulate 3: Fractional Weight Greedy
        sim_cap = cap
        sim_val = 0.0
        for j in order_weight:
            if j == i:
                continue
            if weights[j] <= sim_cap:
                sim_cap -= weights[j]
                sim_val += values[j]
            else:
                if weights[j] > 0:
                    sim_val += values[j] * (sim_cap / weights[j])
                break
        if sim_val > max_sim_val:
            max_sim_val = sim_val

        total_score = base_val + max_sim_val
        
        if total_score > best_score:
            best_score = total_score
            best_idx = i
            
    return int(best_idx)
# EVOLVE-BLOCK-END
