"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 115.0  # safety margin before 120s limit
_start_time = None


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard
    if time.time() - _start_time > _DEADLINE:
        return -1
    
    n = len(remaining_weights)
    if n == 0:
        return -1
    
    w = np.asarray(remaining_weights, dtype=float)
    v = np.asarray(remaining_values, dtype=float)
    c = float(remaining_capacity)
    
    if c <= 0:
        return -1
    
    # Filter to feasible items (weight <= remaining capacity)
    feasible_mask = w <= c
    if not np.any(feasible_mask):
        return -1
    
    feasible_idx = np.where(feasible_mask)[0]
    fw = w[feasible_mask]
    fv = v[feasible_mask]
    
    # For each feasible candidate, estimate total value if we pick it first,
    # then greedily fill remaining capacity with best density items
    best_score = -1.0
    best_candidate = -1
    
    # Precompute densities for all items (for the greedy fill simulation)
    with np.errstate(divide='ignore', invalid='ignore'):
        all_densities = np.where(w > 0, v / w, np.where(v > 0, np.inf, 0.0))
    
    for i in range(len(feasible_idx)):
        orig_idx = feasible_idx[i]
        cand_w = fw[i]
        cand_v = fv[i]
        
        remaining_cap = c - cand_w
        if remaining_cap < 0:
            continue
        
        # Simulate greedy fill of remaining capacity
        # Create a copy of remaining items (excluding the candidate)
        # For efficiency, we work with the full array and mark the candidate as used
        sim_w = w.copy()
        sim_v = v.copy()
        sim_w[orig_idx] = np.inf  # mark as unavailable
        sim_v[orig_idx] = 0.0
        
        # Get feasible items for the simulation
        sim_feasible = sim_w <= remaining_cap
        if not np.any(sim_feasible):
            total_est = cand_v
        else:
            sim_fw = sim_w[sim_feasible]
            sim_fv = sim_v[sim_feasible]
            sim_fidx = np.where(sim_feasible)[0]
            
            with np.errstate(divide='ignore', invalid='ignore'):
                sim_densities = np.where(sim_fw > 0, sim_fv / sim_fw, np.where(sim_fv > 0, np.inf, 0.0))
            
            # Sort by density descending
            order = np.argsort(-sim_densities)
            
            total_est = cand_v
            sim_cap = remaining_cap
            for pos in order:
                item_w = sim_fw[pos]
                item_v = sim_fv[pos]
                if item_w <= sim_cap:
                    total_est += item_v
                    sim_cap -= item_w
                else:
                    break
        
        if total_est > best_score:
            best_score = total_est
            best_candidate = orig_idx
    
    if best_candidate == -1:
        # Fallback: pick highest density feasible item
        with np.errstate(divide='ignore', invalid='ignore'):
            dens = np.where(fw > 0, fv / fw, np.where(fv > 0, np.inf, 0.0))
        best_pos = int(np.argmax(dens))
        return int(feasible_idx[best_pos])
    
    return int(best_candidate)
# EVOLVE-BLOCK-END
