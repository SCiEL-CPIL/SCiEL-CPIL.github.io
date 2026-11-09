"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _simulate_fast(times, pred_masks, cycle_time, priorities):
    n = len(times)
    assigned_mask = 0
    remaining_mask = (1 << n) - 1
    stations = 0
    
    while remaining_mask:
        stations += 1
        current_time = 0.0
        
        while True:
            best_task = -1
            best_prio = -1.0
            
            for i in range(n):
                if not (remaining_mask & (1 << i)):
                    continue
                if (assigned_mask & pred_masks[i]) == pred_masks[i]:
                    if times[i] <= cycle_time - current_time:
                        if priorities[i] > best_prio:
                            best_prio = priorities[i]
                            best_task = i
                            
            if best_task == -1:
                break
                
            assigned_mask |= (1 << best_task)
            remaining_mask &= ~(1 << best_task)
            current_time += times[best_task]
            
    return stations

def priority(instance):
    times = np.asarray(instance["times"], dtype=float)
    preds = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    n = len(times)
    
    # Precompute predecessor bitmasks
    pred_masks = [0] * n
    for i in range(n):
        mask = 0
        for p in preds[i]:
            mask |= (1 << p)
        pred_masks[i] = mask
            
    # Build successors for Rank calculation
    succ = [[] for _ in range(n)]
    for i in range(n):
        for p in preds[i]:
            succ[p].append(i)
            
    # Compute Longest Path to End (Rank)
    memo = [-1.0] * n
    def get_rank(i):
        if memo[i] != -1.0:
            return memo[i]
        if not succ[i]:
            memo[i] = times[i]
            return memo[i]
        max_succ = 0.0
        for s in succ[i]:
            max_succ = max(max_succ, get_rank(s))
        memo[i] = times[i] + max_succ
        return memo[i]
        
    for i in range(n):
        get_rank(i)
        
    # Add small epsilon to break ties consistently with stable sort
    best_priorities = np.array([memo[i] + i * 1e-9 for i in range(n)], dtype=float)
    best_stations = _simulate_fast(times, pred_masks, cycle_time, best_priorities)
    
    # Local Search
    start_time = time.time()
    time_limit = 10.0
    
    rng = np.random.default_rng(42)
    
    while time.time() - start_time < time_limit:
        i = rng.integers(0, n)
        j = rng.integers(0, n)
        if i == j:
            continue
            
        # Swap
        best_priorities[i], best_priorities[j] = best_priorities[j], best_priorities[i]
        
        new_stations = _simulate_fast(times, pred_masks, cycle_time, best_priorities)
        
        if new_stations < best_stations:
            best_stations = new_stations
        else:
            # Revert
            best_priorities[i], best_priorities[j] = best_priorities[j], best_priorities[i]
            
    return best_priorities
# EVOLVE-BLOCK-END
