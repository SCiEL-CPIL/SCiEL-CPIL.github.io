"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations


def construct():
    # Generate D11 roots (norm squared 2)
    roots = []
    for i, j in combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i], v[j] = si, sj
                roots.append(v)
    roots = np.array(roots, dtype=np.int64)
    
    # Generate Norm-4 vectors (4 non-zero entries of +/-1)
    norm4_candidates = []
    for i, j, k, l in combinations(range(D), 4):
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    for sl in (1, -1):
                        v = [0] * D
                        v[i], v[j], v[k], v[l] = si, sj, sk, sl
                        norm4_candidates.append(v)
                        
    norm4_candidates = np.array(norm4_candidates, dtype=np.int64)
    
    # Greedy Independent Set on Norm-4 vectors
    # Condition: min pairwise distance squared >= 4
    selected = []
    for v in norm4_candidates:
        if not selected:
            selected.append(v)
            continue
        
        selected_arr = np.array(selected)
        diffs = selected_arr - v
        dists_sq = np.sum(diffs ** 2, axis=1)
        min_dist_sq = np.min(dists_sq)
        
        if min_dist_sq >= 4:
            selected.append(v)
            
    selected_arr = np.array(selected, dtype=np.int64)
    
    # Compare with roots
    if len(selected_arr) > len(roots):
        return selected_arr
    else:
        return roots
# EVOLVE-BLOCK-END
