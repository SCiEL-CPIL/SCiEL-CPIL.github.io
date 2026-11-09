import numpy as np
import random
import time

N = 64
GRID_SIZE = N * N

def construct():
    start_time = time.time()
    deadline = start_time + 280  # Leave 20s margin for safety
    
    # Precompute coordinates
    coords = np.array([(x, y) for y in range(N) for x in range(N)], dtype=np.int32)
    
    # Precompute squared distances
    # diff shape (4096, 4096, 2)
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2).astype(np.int32)
    del diff
    
    all_indices = list(range(GRID_SIZE))
    best_points_idx = []
    best_size = 0
    
    # Run random restarts until time limit
    seed = 0
    while time.time() < deadline:
        random.seed(seed)
        candidates = all_indices[:]
        random.shuffle(candidates)
        
        selected = []
        
        for p_idx in candidates:
            if not selected:
                selected.append(p_idx)
                continue
                
            valid = True
            
            # 1. Check if P is apex (d(P, A) == d(P, B))
            d_p_s = dist_sq[p_idx, selected]
            if len(np.unique(d_p_s)) != len(d_p_s):
                valid = False
            else:
                # 2. Check if A or B is apex
                # d(P, A) == d(A, B) or d(P, B) == d(A, B)
                # D_sel[i, j] = d(selected[i], selected[j])
                # V_p[i] = d(p, selected[i])
                # We check if V_p[i] == D_sel[i, j] for any j
                D_sel = dist_sq[selected, selected]
                V_p = d_p_s[:, np.newaxis]
                M = (V_p == D_sel)
                if np.any(M):
                    valid = False
            
            if valid:
                selected.append(p_idx)
                
        if len(selected) > best_size:
            best_size = len(selected)
            best_points_idx = selected[:]
            # Optional: break if we reach a theoretical upper bound? 
            # Hard to know. Just keep improving.
            
        seed += 1
        
    return [tuple(coords[i]) for i in best_points_idx]
