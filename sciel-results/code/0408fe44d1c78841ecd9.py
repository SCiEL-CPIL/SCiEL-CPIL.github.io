"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard: evaluator kills after 150s. We keep a safety margin.
_START_TIME = time.time()
_DEADLINE_SECONDS = 140.0  # 10s safety margin before the 150s kill

def _elapsed_ok():
    return (time.time() - _START_TIME) < _DEADLINE_SECONDS

def _fractional_knapsack_bound(capacity, weights, values, exclude_idx=-1):
    """
    Compute the fractional knapsack upper bound for the given items.
    Sort by density descending, take items fully until capacity is exhausted,
    then take a fraction of the next item.
    Returns the bound value.
    """
    n = len(weights)
    if n == 0 or capacity <= 0:
        return 0.0
    
    # Filter to feasible items (weight <= capacity) and exclude the excluded index
    feasible = []
    for i in range(n):
        if i == exclude_idx:
            continue
        if weights[i] <= capacity:
            feasible.append(i)
    
    if not feasible:
        return 0.0
    
    # Sort by density descending
    # Use a stable sort; break ties by index for determinism
    def density_key(i):
        if weights[i] > 0:
            return values[i] / weights[i]
        return float('inf')
    
    feasible.sort(key=lambda i: (-density_key(i), i))
    
    cap = capacity
    total_val = 0.0
    for i in feasible:
        if cap <= 0:
            break
        w_i = weights[i]
        v_i = values[i]
        if w_i <= cap:
            cap -= w_i
            total_val += v_i
        else:
            # Take fraction
            total_val += v_i * (cap / w_i)
            cap = 0.0
    
    return total_val

def _greedy_fill(capacity, weights, values, order_indices, exclude_idx=-1):
    """
    Simulate greedy knapsack fill using the given order of indices.
    Returns the total value achieved.
    """
    sim_cap = capacity
    sim_val = 0.0
    for idx in order_indices:
        if idx == exclude_idx:
            continue
        w = weights[idx]
        if w <= sim_cap:
            sim_cap -= w
            sim_val += values[idx]
        if sim_cap <= 0:
            break
    return sim_val

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    # Safety check: if we've been running too long, fall back to simple greedy
    if not _elapsed_ok():
        # Fallback: simple density greedy
        if len(remaining_weights) == 0 or remaining_capacity <= 0:
            return -1
        weights = np.asarray(remaining_weights, dtype=float)
        values = np.asarray(remaining_values, dtype=float)
        feasible = np.where(weights <= remaining_capacity)[0]
        if len(feasible) == 0:
            return -1
        # Pick best density among feasible
        with np.errstate(divide='ignore', invalid='ignore'):
            dens = np.where(weights[feasible] > 0, values[feasible] / weights[feasible], 0.0)
        best_local = np.argmax(dens)
        return int(feasible[best_local])

    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1

    weights = np.asarray(remaining_weights, dtype=float)
    values = np.asarray(remaining_values, dtype=float)
    c = float(remaining_capacity)
    
    # Filter to feasible items
    feasible_mask = weights <= c
    if not np.any(feasible_mask):
        return -1
    
    feasible_indices = np.where(feasible_mask)[0]
    
    if len(feasible_indices) == 1:
        return int(feasible_indices[0])
    
    # Pre-compute sorting orders for multiple greedy strategies
    with np.errstate(divide='ignore', invalid='ignore'):
        densities = np.where(weights > 0, values / weights, 0.0)
    
    order_density = np.argsort(-densities)
    order_value = np.argsort(-values)
    
    # Compute the fractional knapsack bound for the current state (excluding nothing)
    # This serves as the global upper bound
    global_bound = _fractional_knapsack_bound(c, weights, values)
    
    best_total_val = -1.0
    best_idx = -1
    
    # For each feasible item, compute a composite score:
    # 1. Immediate value + greedy future (density order)
    # 2. Immediate value + greedy future (value order)
    # 3. Surrogate LP gap: how much of the global bound is "wasted" by picking this item
    #    We approximate the post-pick bound and compare to global_bound
    #    A smaller gap means the pick is closer to the LP optimum
    
    for idx in feasible_indices:
        if not _elapsed_ok():
            break
            
        w_i = weights[idx]
        v_i = values[idx]
        remaining_cap = c - w_i
        
        if remaining_cap < 0:
            continue
        
        # Greedy simulation 1: density order
        val_sim_1 = _greedy_fill(remaining_cap, weights, values, order_density, idx)
        
        # Greedy simulation 2: value order
        val_sim_2 = _greedy_fill(remaining_cap, weights, values, order_value, idx)
        
        # Best greedy future
        best_greedy_future = max(val_sim_1, val_sim_2)
        
        # Compute the fractional bound after picking this item
        # This is the LP relaxation of the remaining subproblem
        post_bound = _fractional_knapsack_bound(remaining_cap, weights, values, idx)
        
        # The "gap" is how much value we leave on the table relative to the LP bound
        # We want to minimize the gap, i.e., maximize (post_bound - best_greedy_future) is bad
        # Actually, we want the total value to be high. 
        # Let's use a composite score:
        # score = v_i + best_greedy_future + alpha * (post_bound - (v_i + best_greedy_future))
        # This blends the achievable greedy value with the LP potential.
        # If the greedy fill is close to the LP bound, the gap is small, so the score is close to v_i + best_greedy_future.
        # If the greedy fill is far from the LP bound, the score is boosted by the potential, encouraging items that open up better LP solutions.
        
        # A simpler and more robust approach: 
        # score = v_i + best_greedy_future
        # But we also want to prefer items that don't "waste" capacity on low-density items.
        # The LP bound itself is a strong signal. Let's use:
        # score = v_i + 0.5 * best_greedy_future + 0.5 * post_bound
        
        # Actually, let's try a weighted combination that emphasizes the LP bound gap minimization.
        # The idea: pick the item that minimizes the gap between the post-pick LP bound and the post-pick greedy value.
        # gap = post_bound - (v_i + best_greedy_future)
        # We want to minimize gap, but also maximize v_i + best_greedy_future.
        
        # Let's use: score = (v_i + best_greedy_future) - 0.1 * gap
        # This penalizes items that lead to large gaps between greedy and LP.
        
        gap = post_bound - (v_i + best_greedy_future)
        if gap < 0:
            gap = 0.0  # Shouldn't happen, but just in case
        
        score = (v_i + best_greedy_future) - 0.1 * gap
        
        if score > best_total_val:
            best_total_val = score
            best_idx = int(idx)
    
    if best_idx == -1:
        return -1
        
    return best_idx
# EVOLVE-BLOCK-END
