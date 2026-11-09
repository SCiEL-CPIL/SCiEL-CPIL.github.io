# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations

def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional points with integral coordinates.
    Uses a multi-weight greedy approach: first select norm-2 vectors (sparser conflicts),
    then norm-3, then norm-4 vectors to fill remaining space.
    Condition: min_dist^2 >= max_norm^2.
    """
    start_time = time.time()
    deadline = start_time + 600
    
    d = 11
    
    # Generate candidates by weight (number of non-zero entries)
    # Weight 2: norm^2 = 2, 220 vectors
    # Weight 3: norm^2 = 3, 1320 vectors  
    # Weight 4: norm^2 = 4, 5280 vectors
    # Weight 5: norm^2 = 5, 15840 vectors (might be too many)
    
    candidates_by_weight = {}
    
    # Weight 2: ±e_i ± e_j
    cands_w2 = []
    for i, j in combinations(range(d), 2):
        for si in [1, -1]:
            for sj in [1, -1]:
                p = np.zeros(d, dtype=np.int64)
                p[i] = si
                p[j] = sj
                cands_w2.append(p)
    candidates_by_weight[2] = np.array(cands_w2, dtype=np.int64)
    
    # Weight 3: ±e_i ± e_j ± e_k
    cands_w3 = []
    for i, j, k in combinations(range(d), 3):
        for si in [1, -1]:
            for sj in [1, -1]:
                for sk in [1, -1]:
                    p = np.zeros(d, dtype=np.int64)
                    p[i] = si
                    p[j] = sj
                    p[k] = sk
                    cands_w3.append(p)
    candidates_by_weight[3] = np.array(cands_w3, dtype=np.int64)
    
    # Weight 4: ±e_i ± e_j ± e_k ± e_l
    cands_w4 = []
    for i, j, k, l in combinations(range(d), 4):
        for si in [1, -1]:
            for sj in [1, -1]:
                for sk in [1, -1]:
                    for sl in [1, -1]:
                        p = np.zeros(d, dtype=np.int64)
                        p[i] = si
                        p[j] = sj
                        p[k] = sk
                        p[l] = sl
                        cands_w4.append(p)
    candidates_by_weight[4] = np.array(cands_w4, dtype=np.int64)
    
    # Greedy selection with multi-weight strategy
    # Start with weight 2 (smallest norm, sparsest conflicts)
    # Then weight 3, then weight 4
    # For each weight, use random shuffle to break symmetries
    
    rng = np.random.default_rng(42)
    selected = []
    
    for weight in [2, 3, 4]:
        if time.time() > deadline - 30:
            break
            
        cands = candidates_by_weight[weight]
        n_cands = len(cands)
        perm = rng.permutation(n_cands)
        
        # For this weight, greedily add valid candidates
        for idx in perm:
            if time.time() > deadline - 10:
                break
                
            v = cands[idx]
            valid = True
            for u in selected:
                dot = np.dot(v, u)
                # Conflict if dist^2 < max_norm^2
                # dist^2 = |v|^2 + |u|^2 - 2*dot
                # max_norm^2 = max(|v|^2, |u|^2)
                # We need: |v|^2 + |u|^2 - 2*dot >= max(|v|^2, |u|^2)
                # Let's compute properly:
                norm_v_sq = weight * weight  # Wait, no. Weight is number of nonzeros, each is ±1, so norm^2 = weight
                norm_u_sq = np.dot(u, u)
                dist_sq = norm_v_sq + norm_u_sq - 2 * dot
                max_norm_sq = max(norm_v_sq, norm_u_sq)
                if dist_sq < max_norm_sq:
                    valid = False
                    break
            
            if valid:
                selected.append(v)
    
    if not selected:
        # Fallback
        points_list = []
        for i, j in combinations(range(d), 2):
            for si in [1, -1]:
                for sj in [1, -1]:
                    p = np.zeros(d, dtype=np.int64)
                    p[i] = si
                    p[j] = sj
                    points_list.append(p)
        return np.array(points_list, dtype=np.int64)
        
    points = np.array(selected, dtype=np.int64)
    return points

# EVOLVE-BLOCK-END
