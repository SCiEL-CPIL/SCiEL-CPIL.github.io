import time
import numpy as np
import random
from collections import Counter

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin

    def get_shape_factors(indices):
        n = len(indices)
        if n != 30:
            return Counter()
        
        counts = Counter()
        # indices is a list of (u, v) tuples
        for i in range(n):
            u1, v1 = indices[i]
            for j in range(i + 1, n):
                u2, v2 = indices[j]
                for k in range(j + 1, n):
                    u3, v3 = indices[k]
                    
                    du1 = u2 - u1
                    dv1 = v2 - v1
                    du2 = u3 - u1
                    dv2 = v3 - v1
                    
                    s = abs(du1 * dv2 - du2 * dv1)
                    if s > 0:
                        counts[s] += 1
                        
        return counts

    def get_max_multiplicity(counts):
        if not counts:
            return 0
        return max(counts.values())

    def evaluate_index_config(indices):
        counts = get_shape_factors(indices)
        return get_max_multiplicity(counts)

    def local_search(indices, time_limit):
        current_indices = list(indices)
        current_score = evaluate_index_config(current_indices)
        best_indices = list(current_indices)
        best_score = current_score
        
        current_set = set(current_indices)
        no_improve = 0
        max_no_improve = 500
        
        while time.time() < time_limit:
            if no_improve > max_no_improve:
                break
                
            idx = random.randint(0, 29)
            old_u, old_v = current_indices[idx]
            
            # Adaptive move size
            move_size = 1 if no_improve < 200 else 2
            
            new_u = old_u + random.randint(-move_size, move_size)
            new_v = old_v + random.randint(-move_size, move_size)
            
            # Constrain to a box to keep it compact
            # A 5x6 grid fits in 5x6. Let's allow [-5, 5] x [-5, 5]
            if not (-5 <= new_u <= 5 and -5 <= new_v <= 5):
                continue
                
            new_point = (new_u, new_v)
            
            if new_point == old_point:
                continue
                
            if new_point in current_set:
                continue
                
            current_indices[idx] = new_point
            current_set.remove(old_point)
            current_set.add(new_point)
            
            new_score = evaluate_index_config(current_indices)
            
            if new_score > current_score:
                current_score = new_score
                no_improve = 0
                if current_score > best_score:
                    best_score = current_score
                    best_indices = list(current_indices)
            else:
                current_indices[idx] = old_point
                current_set.remove(new_point)
                current_set.add(old_point)
                no_improve += 1
                
        return best_indices, best_score

    # Define several starting configurations
    starters = []
    
    # 5x6 Grid
    starters.append([(u, v) for u in range(5) for v in range(6)])
    # 6x5 Grid
    starters.append([(u, v) for u in range(6) for v in range(5)])
    
    # 4x8 Grid minus 2 points (to make 30)
    # 4x8 = 32. Remove 2.
    grid_4x8 = [(u, v) for u in range(4) for v in range(8)]
    # Remove corners?
    starters.append(grid_4x8[:30])
    
    # 5x5 Grid plus 5 points
    grid_5x5 = [(u, v) for u in range(5) for v in range(5)]
    # Add 5 points in a row next to it?
    extra = [(5, v) for v in range(5)]
    starters.append(grid_5x5 + extra)
    
    best_overall_indices = None
    best_overall_score = 0
    
    # Run local search on each starter with a time budget
    time_per_starter = (deadline - time.time() - 2.0) / len(starters)
    
    for i, starter in enumerate(starters):
        if time.time() > deadline - 2.0:
            break
            
        t_start = time.time()
        t_end = min(t_start + time_per_starter, deadline - 2.0)
        
        indices, score = local_search(starter, t_end)
        
        if score > best_overall_score:
            best_overall_score = score
            best_overall_indices = indices

    if best_overall_indices is None:
        # Fallback
        best_overall_indices = starters[0]

    # Map to physical coordinates
    us = [p[0] for p in best_overall_indices]
    vs = [p[1] for p in best_overall_indices]
    
    min_u, max_u = min(us), max(us)
    min_v, max_v = min(vs), max(vs)
    
    offset_u = -(min_u + max_u) // 2
    offset_v = -(min_v + max_v) // 2
    
    physical_points = [(u + offset_u, v + offset_v) for u, v in best_overall_indices]
    
    # Final check
    if len(set(physical_points)) != 30:
        # Should not happen, but fallback
        physical_points = [(u, v) for u in range(5) for v in range(6)]
        min_u, max_u = 0, 4
        min_v, max_v = 0, 5
        offset_u = -(min_u + max_u) // 2
        offset_v = -(min_v + max_v) // 2
        physical_points = [(u + offset_u, v + offset_v) for u, v in physical_points]

    return physical_points
