import time
import numpy as np
import random

def _fast_score(points, i_idx, j_idx, k_idx):
    p1 = points[i_idx]
    p2 = points[j_idx]
    p3 = points[k_idx]
    det = np.abs(p1[:, 0] * (p2[:, 1] - p3[:, 1]) + 
                 p2[:, 0] * (p3[:, 1] - p1[:, 1]) + 
                 p3[:, 0] * (p1[:, 1] - p2[:, 1]))
    pos = det[det > 0]
    if len(pos) == 0: return 0
    _, counts = np.unique(pos, return_counts=True)
    return int(np.max(counts))

def _sa_search(points_init, time_limit, i_idx, j_idx, k_idx):
    n = 30
    points = np.array(points_init, dtype=np.int32)
    occupied = set(map(tuple, points))
    
    current_score = _fast_score(points, i_idx, j_idx, k_idx)
    best_score = current_score
    best_points = points.copy()
    
    T = 10.0
    T_min = 0.1
    cooling = 0.9999
    
    start_time = time.time()
    
    while time.time() < start_time + time_limit:
        idx = random.randint(0, n - 1)
        old_x, old_y = points[idx]
        
        # Move generation
        if random.random() < 0.7:
            # Local
            new_x = old_x + random.randint(-4, 4)
            new_y = old_y + random.randint(-4, 4)
        else:
            # Global
            new_x = random.randint(-30, 30)
            new_y = random.randint(-30, 30)
            
        new_x = max(-30, min(30, new_x))
        new_y = max(-30, min(30, new_y))
        
        new_p = (new_x, new_y)
        if new_p in occupied:
            continue
            
        occupied.remove((old_x, old_y))
        occupied.add(new_p)
        points[idx] = [new_x, new_y]
        
        new_score = _fast_score(points, i_idx, j_idx, k_idx)
        
        if new_score >= current_score:
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_points = points.copy()
        else:
            delta = current_score - new_score
            if T > T_min:
                prob = np.exp(-delta / T)
                if random.random() < prob:
                    current_score = new_score
                else:
                    # Revert
                    occupied.remove(new_p)
                    occupied.add((old_x, old_y))
                    points[idx] = [old_x, old_y]
        
        T = max(T_min, T * cooling)
        
    return best_points, best_score

def construct():
    start_time = time.time()
    total_budget = 55.0
    
    # Precompute triple indices
    n = 30
    i_list, j_list, k_list = [], [], []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                i_list.append(i)
                j_list.append(j)
                k_list.append(k)
    i_idx = np.array(i_list)
    j_idx = np.array(j_list)
    k_idx = np.array(k_list)
    
    best_points_global = None
    best_score_global = 0
    
    # Define initial grids
    grids = []
    # 6x5
    grids.append([(-2 + i, -2 + j) for i in range(6) for j in range(5)])
    # 5x6
    grids.append([(-2 + i, -2 + j) for i in range(5) for j in range(6)])
    # 3x10
    grids.append([(-4 + i, -4 + j) for i in range(3) for j in range(10)])
    # 2x15
    grids.append([(-6 + i, -6 + j) for i in range(2) for j in range(15)])
    
    # Allocate time
    # 10s per grid, 10s for polish
    time_per_phase = 10.0
    
    for g in grids:
        if time.time() > start_time + total_budget - 10:
            break
        pts, score = _sa_search(g, time_per_phase, i_idx, j_idx, k_idx)
        if score > best_score_global:
            best_score_global = score
            best_points_global = pts
            
    # Polish phase with remaining time
    remaining = total_budget - (time.time() - start_time)
    if remaining > 2 and best_points_global is not None:
        # Run SA on the best points found so far
        pts, score = _sa_search(best_points_global, remaining - 2, i_idx, j_idx, k_idx)
        if score > best_score_global:
            best_score_global = score
            best_points_global = pts
            
    if best_points_global is None:
        # Fallback
        return [(-2 + i, -2 + j) for i in range(6) for j in range(5)]
        
    return [tuple(p) for p in best_points_global]
