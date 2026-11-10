# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
import random

def kissing_number11() -> np.ndarray:
    """
    Finds a large set of integer points in Z^11 such that min pairwise squared distance
    >= max squared norm. Uses random sampling + local optimization.
    """
    start_time = time.time()
    DEADLINE = 590  # safety margin before 630s kill
    d = 11
    
    def elapsed():
        return time.time() - start_time
    
    def remaining():
        return DEADLINE - elapsed()
    
    # Generate a diverse pool of candidate points
    # Points with small coordinates to keep norms bounded
    candidates = []
    
    # Norm squared = 1: single ±1
    for i in range(d):
        for s in [1, -1]:
            p = np.zeros(d, dtype=np.int64)
            p[i] = s
            candidates.append(p)
    
    # Norm squared = 2: two ±1
    for i, j in itertools.combinations(range(d), 2):
        for si in [1, -1]:
            for sj in [1, -1]:
                p = np.zeros(d, dtype=np.int64)
                p[i] = si
                p[j] = sj
                candidates.append(p)
    
    # Norm squared = 3: three ±1
    for i, j, k in itertools.combinations(range(d), 3):
        for si in [1, -1]:
            for sj in [1, -1]:
                for sk in [1, -1]:
                    p = np.zeros(d, dtype=np.int64)
                    p[i] = si
                    p[j] = sj
                    p[k] = sk
                    candidates.append(p)
    
    # Norm squared = 4: four ±1, or one ±2
    for i, j, k, l in itertools.combinations(range(d), 4):
        for si in [1, -1]:
            for sj in [1, -1]:
                for sk in [1, -1]:
                    for sl in [1, -1]:
                        p = np.zeros(d, dtype=np.int64)
                        p[i] = si
                        p[j] = sj
                        p[k] = sk
                        p[l] = sl
                        candidates.append(p)
    for i in range(d):
        for s in [2, -2]:
            p = np.zeros(d, dtype=np.int64)
            p[i] = s
            candidates.append(p)
    
    # Norm squared = 5: five ±1, or (±2,±1), or (±1,±1,±1,±1,±1)
    for combo in itertools.combinations(range(d), 5):
        for signs in itertools.product([1, -1], repeat=5):
            p = np.zeros(d, dtype=np.int64)
            for idx, val in zip(combo, signs):
                p[idx] = val
            candidates.append(p)
    for i, j in itertools.combinations(range(d), 2):
        for si in [2, -2]:
            for sj in [1, -1]:
                p = np.zeros(d, dtype=np.int64)
                p[i] = si
                p[j] = sj
                candidates.append(p)
    
    cand_arr = np.array(candidates, dtype=np.int64)
    n_cand = len(cand_arr)
    norms_sq = np.sum(cand_arr ** 2, axis=1)
    
    best_points = None
    best_count = 0
    best_R = 0
    
    # Strategy: For each possible R (max norm squared), find max compatible set
    # where all points have norm_sq <= R and pairwise dist_sq > R
    
    for R in range(1, 8):
        if remaining() < 30:
            break
        
        mask = norms_sq <= R
        filtered = cand_arr[mask]
        n_filt = len(filtered)
        
        if n_filt < 2:
            continue
        
        # Greedy with random restarts
        for restart in range(50):
            if remaining() < 15:
                break
            
            # Random ordering for diversity
            if restart == 0:
                order = np.arange(n_filt)
            else:
                order = np.random.permutation(n_filt)
            
            selected = []
            selected_arr = []
            
            for idx in order:
                if remaining() < 15:
                    break
                
                p = filtered[idx]
                
                # Check compatibility with all selected
                if not selected:
                    selected.append(idx)
                    selected_arr.append(p)
                    continue
                
                # Vectorized distance check
                diffs = np.array(selected_arr) - p
                dists_sq = np.sum(diffs ** 2, axis=1)
                
                if np.min(dists_sq) > R:
                    selected.append(idx)
                    selected_arr.append(p)
            
            if len(selected) > best_count:
                best_count = len(selected)
                best_points = np.array(selected_arr, dtype=np.int64)
                best_R = R
    
    # Local optimization: try to add more points or swap to increase count
    if best_points is not None and best_count >= 2:
        # Try to add points from candidates not in the set
        current_set = set(map(tuple, best_points.tolist()))
        for R in range(1, 8):
            if remaining() < 10:
                break
            
            mask = norms_sq <= R
            filtered = cand_arr[mask]
            
            for idx in range(len(filtered)):
                if remaining() < 10:
                    break
                
                p = filtered[idx]
                if tuple(p.tolist()) in current_set:
                    continue
                
                # Check if p can be added
                diffs = best_points - p
                dists_sq = np.sum(diffs ** 2, axis=1)
                
                if len(dists_sq) == 0 or np.min(dists_sq) > R:
                    # Also check norm constraint
                    if np.dot(p, p) <= R:
                        best_points = np.vstack([best_points, p])
                        current_set.add(tuple(p.tolist()))
                        best_count = len(best_points)
    
    # Final fallback
    if best_points is None or len(best_points) < 2:
        best_points = np.array([np.ones(d, dtype=np.int64), -np.ones(d, dtype=np.int64)])
    
    # Validate no zero vectors
    for i in range(len(best_points)):
        if np.all(best_points[i] == 0):
            best_points = np.array([np.ones(d, dtype=np.int64), -np.ones(d, dtype=np.int64)])
            break
    
    return best_points

# EVOLVE-BLOCK-END
