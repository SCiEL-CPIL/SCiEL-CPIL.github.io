"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global start time for safety
_start_time = time.time()
_TIMEOUT = 145.0  # Safety margin before 150s kill

def _fractional_knapsack_value(order, weights, values, cap, skip_idx):
    """Compute fractional knapsack value using a given order, skipping one item."""
    sim_cap = cap
    sim_val = 0.0
    for j in order:
        if j == skip_idx:
            continue
        if weights[j] <= sim_cap:
            sim_cap -= weights[j]
            sim_val += values[j]
        else:
            if weights[j] > 0 and sim_cap > 0:
                sim_val += values[j] * (sim_cap / weights[j])
            break
    return sim_val

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
    with np.errstate(divide='ignore', invalid='ignore'):
        densities = values / weights
    order_density = np.argsort(-densities)
    order_value = np.argsort(-values)
    order_weight = np.argsort(weights)
    
    # Combined ordering: density * 0.6 + normalized_value * 0.4
    if n > 0:
        max_val = np.max(values) if np.max(values) > 0 else 1.0
        min_val = np.min(values) if np.min(values) < 0 else 0.0
        val_range = max_val - min_val
        if val_range > 0:
            norm_vals = (values - min_val) / val_range
        else:
            norm_vals = np.ones(n)
        combined_score = 0.6 * (densities / (np.max(densities) + 1e-10)) + 0.4 * norm_vals
        order_combined = np.argsort(-combined_score)
    else:
        order_combined = np.array([], dtype=int)
    
    best_score = -1.0
    best_idx = -1
    
    # Soft-max temperature: lower = more like hard max, higher = smoother
    T = 5.0
    
    for i in range(n):
        if weights[i] > remaining_capacity:
            continue
            
        cap = remaining_capacity - weights[i]
        base_val = values[i]
        
        # Compute values from each simulation
        sim1 = _fractional_knapsack_value(order_density, weights, values, cap, i)
        sim2 = _fractional_knapsack_value(order_value, weights, values, cap, i)
        sim3 = _fractional_knapsack_value(order_weight, weights, values, cap, i)
        sim4 = _fractional_knapsack_value(order_combined, weights, values, cap, i)
        
        # Soft-max (log-sum-exp) to smoothly combine simulation results
        # This avoids the nonsmooth hard-max and weights all strategies
        sims = np.array([sim1, sim2, sim3, sim4], dtype=float)
        sims_scaled = sims / T
        max_sim = np.max(sims_scaled)
        log_sum_exp = max_sim + np.log(np.sum(np.exp(sims_scaled - max_sim)))
        soft_max_val = T * log_sum_exp
        
        total_score = base_val + soft_max_val
        
        if total_score > best_score:
            best_score = total_score
            best_idx = i
            
    return int(best_idx)
# EVOLVE-BLOCK-END
