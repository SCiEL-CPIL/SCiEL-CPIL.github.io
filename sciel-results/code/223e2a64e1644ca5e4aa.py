"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global start time for safety
_START_TIME = time.time()
_DEADLINE_SECONDS = 120.0
_SAFETY_MARGIN = 2.0

def _elapsed_ok():
    return (time.time() - _START_TIME) < (_DEADLINE_SECONDS - _SAFETY_MARGIN)

def _vectorized_greedy_fill(capacity, weights, values, order_indices, exclude_mask):
    """
    Vectorized greedy fill simulation.
    Given a capacity and an ordering of items, simulate picking items greedily.
    Returns the total value from the fill.
    
    We use a simple iterative approach but vectorize the feasibility check.
    Since we need to stop when capacity runs out, we process in chunks.
    """
    if capacity <= 0:
        return 0.0
    
    # Create a mask for items that are not excluded
    available = ~exclude_mask
    
    # Process items in the given order
    sim_cap = capacity
    sim_val = 0.0
    
    # We'll process in batches for efficiency
    processed = np.zeros(len(order_indices), dtype=bool)
    
    for idx in order_indices:
        if processed[idx]:
            continue
        if not available[idx]:
            continue
        w = weights[idx]
        if w <= sim_cap:
            sim_cap -= w
            sim_val += values[idx]
            processed[idx] = True
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
        dens = values[feasible] / np.maximum(weights[feasible], 1e-10)
        best_local = np.argmax(dens)
        return int(feasible[best_local])

    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1

    weights = np.asarray(remaining_weights, dtype=float)
    values = np.asarray(remaining_values, dtype=float)
    
    # Filter to feasible items
    feasible_mask = weights <= remaining_capacity
    if not np.any(feasible_mask):
        return -1
    
    feasible_indices = np.where(feasible_mask)[0]
    n_feasible = len(feasible_indices)
    
    if n_feasible == 1:
        return int(feasible_indices[0])

    # Pre-compute sorting orders for multiple different greedy strategies
    # 1. Density (Value / Weight)
    with np.errstate(divide='ignore', invalid='ignore'):
        densities = np.where(weights > 0, values / weights, 0.0)
    order_density = np.argsort(-densities)
    
    # 2. Value (Highest Value first)
    order_value = np.argsort(-values)
    
    # 3. Weight (Lightest first)
    order_weight = np.argsort(weights)
    
    # 4. Ratio: Value / (Weight + epsilon) - slightly different density
    with np.errstate(divide='ignore', invalid='ignore'):
        densities2 = values / (weights + 0.1)
    order_density2 = np.argsort(-densities2)
    
    # 5. Score: Value * sqrt(1/Weight) - another heuristic
    with np.errstate(divide='ignore', invalid='ignore'):
        scores = values * np.sqrt(np.where(weights > 0, 1.0/weights, 0.0))
    order_score = np.argsort(-scores)
    
    # Store all orders
    orders = [order_density, order_value, order_weight, order_density2, order_score]
    n_orders = len(orders)
    
    # Pre-compute for each order, the prefix sums of weights and values
    # This allows us to quickly estimate greedy fill
    prefix_weights = []
    prefix_values = []
    for order in orders:
        pw = np.cumsum(weights[order])
        pv = np.cumsum(values[order])
        prefix_weights.append(pw)
        prefix_values.append(pv)
    
    best_total_val = -1.0
    best_idx = -1
    
    # For each feasible item, estimate the total value if we pick it now
    # and then fill the rest with the best greedy strategy
    for idx in feasible_indices:
        if not _elapsed_ok():
            break
            
        w_i = weights[idx]
        v_i = values[idx]
        remaining_cap = remaining_capacity - w_i
        
        if remaining_cap <= 0:
            # No capacity left after picking this item
            total_val = v_i
        else:
            # Simulate fill with each strategy, take the max
            best_fill_val = 0.0
            
            for order_idx, order in enumerate(orders):
                # Create exclude mask
                exclude_mask = np.zeros(n, dtype=bool)
                exclude_mask[idx] = True
                
                # Simulate greedy fill
                sim_cap = remaining_cap
                sim_val = 0.0
                
                for item_idx in order:
                    if exclude_mask[item_idx]:
                        continue
                    w = weights[item_idx]
                    if w <= sim_cap:
                        sim_cap -= w
                        sim_val += values[item_idx]
                    if sim_cap <= 0:
                        break
                
                if sim_val > best_fill_val:
                    best_fill_val = sim_val
            
            total_val = v_i + best_fill_val
        
        if total_val > best_total_val:
            best_total_val = total_val
            best_idx = idx
            
    if best_idx == -1:
        return -1
        
    return int(best_idx)
# EVOLVE-BLOCK-END
