import time
import random
import numpy as np

N = 64

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Safety margin
    
    # Precompute all points
    all_points = [(x, y) for y in range(N) for x in range(N)]
    n_points = len(all_points)
    
    # Precompute squared distance matrix
    coords = np.array(all_points, dtype=np.float64)
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist_mat = np.sum(diff * diff, axis=2)
    
    best_set = []
    
    for restart in range(1000):
        if time.time() > deadline:
            break
        
        random.seed(restart)
        order = list(range(n_points))
        random.shuffle(order)
        
        # Precompute rank for this order
        rank = np.zeros(n_points, dtype=int)
        for idx, val in enumerate(order):
            rank[val] = idx
        
        S = []
        S_arr = np.array(S, dtype=int)
        
        for p in order:
            if time.time() > deadline:
                break
            
            if len(S) == 0:
                S.append(p)
                S_arr = np.array(S, dtype=int)
                continue
            
            # d_all_S: distances from all points to S
            d_all_S = dist_mat[:, S_arr]
            
            # Check 1: Unique distances from P to S
            d_sorted = np.sort(d_all_S, axis=1)
            unique_ok = np.all(d_sorted[:, 1:] != d_sorted[:, :-1], axis=1)
            
            # Check 2: No conflict with existing distances in S
            D_S = dist_mat[np.ix_(S_arr, S_arr)]
            
            conflict = np.zeros(n_points, dtype=bool)
            for j in range(len(S)):
                vals = d_all_S[:, j]
                ref = D_S[j, :]
                is_in = np.isin(vals, ref)
                conflict |= is_in
            
            valid = unique_ok & ~conflict
            
            # Exclude points already in S
            mask = np.ones(n_points, dtype=bool)
            mask[S_arr] = False
            valid &= mask
            
            if np.any(valid):
                valid_indices = np.where(valid)[0]
                if len(valid_indices) > 0:
                    valid_ranks = rank[valid_indices]
                    min_rank_idx = np.argmin(valid_ranks)
                    chosen = valid_indices[min_rank_idx]
                    
                    S.append(chosen)
                    S_arr = np.array(S, dtype=int)
            else:
                break
        
        if len(S) > len(best_set):
            best_set = S.copy()
    
    return [all_points[i] for i in best_set]
