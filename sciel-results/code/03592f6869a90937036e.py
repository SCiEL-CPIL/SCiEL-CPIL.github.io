import numpy as np
import random
import time
import math

# Precompute all 4-term arithmetic progressions in 0..127
_ap_indices_list = []
for d in range(1, 43):
    max_a = 127 - 3 * d
    for a in range(max_a + 1):
        _ap_indices_list.append((a, a+d, a+2*d, a+3*d))

_AP_INDICES = np.array(_ap_indices_list)

def _get_cost(colors):
    """
    Calculate cost: violations * 10000 + num_colors_used
    Returns int.
    """
    vals = colors[_AP_INDICES]
    sorted_vals = np.sort(vals, axis=1)
    diffs = sorted_vals[:, 1:] != sorted_vals[:, :-1]
    distinct_counts = 1 + np.sum(diffs, axis=1)
    violations = np.sum(distinct_counts < 3)
    
    if violations > 0:
        return int(violations * 10000)
    
    num_colors = len(np.unique(colors))
    return int(num_colors)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0 # Safety margin
    
    n = 128
    # Start with 4 colors
    colors = np.random.randint(0, 4, size=n)
    
    current_cost = _get_cost(colors)
    best_cost = current_cost
    best_colors = colors.copy()
    
    # If we start lucky, return immediately
    if best_cost == 3:
        return list(best_colors)
        
    # Simulated Annealing
    T = 100.0
    T_min = 0.01
    cooling_rate = 0.999
    
    while T > T_min:
        if time.time() > deadline:
            break
            
        # Batch of 100 iterations
        for _ in range(100):
            if time.time() > deadline:
                break
                
            i = np.random.randint(0, n)
            old_color = colors[i]
            new_color = np.random.randint(0, 4)
            
            if new_color == old_color:
                continue
                
            colors[i] = new_color
            new_cost = _get_cost(colors)
            
            delta = new_cost - current_cost
            
            # Accept if better or with probability exp(-delta/T)
            if delta < 0 or (T > 0 and random.random() < math.exp(-delta / T)):
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_colors = colors.copy()
                    if best_cost == 3:
                        # Optimal found
                        return list(best_colors)
            else:
                colors[i] = old_color
                
        T *= cooling_rate
        
    # Post-processing: Try to reduce colors if we have a valid solution
    if best_cost < 10000 and best_cost > 3:
        unique_colors = np.unique(best_colors)
        # Try to merge any color into another
        for c1 in unique_colors:
            for c2 in unique_colors:
                if c1 == c2:
                    continue
                new_colors = best_colors.copy()
                new_colors[new_colors == c1] = c2
                new_cost = _get_cost(new_colors)
                if new_cost == 3:
                    return list(new_colors)
                elif new_cost < best_cost:
                    best_colors = new_colors
                    best_cost = new_cost
                    if best_cost == 3:
                        return list(best_colors)

    return list(best_colors)
