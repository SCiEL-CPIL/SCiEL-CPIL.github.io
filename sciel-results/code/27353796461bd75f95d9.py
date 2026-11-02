"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5 second safety margin
    
    # Generate candidates: all vectors with entries in {-1, 0, 1} and weight 1-4
    candidates = []
    
    # Weight 1
    for i in range(D):
        for s in (1, -1):
            v = np.zeros(D, dtype=np.int64)
            v[i] = s
            candidates.append(v)
    
    # Weight 2
    for i, j in combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i], v[j] = si, sj
                candidates.append(v)
    
    # Weight 3
    for i, j, k in combinations(range(D), 3):
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    v = np.zeros(D, dtype=np.int64)
                    v[i], v[j], v[k] = si, sj, sk
                    candidates.append(v)
    
    # Weight 4
    for i, j, k, l in combinations(range(D), 4):
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    for sl in (1, -1):
                        v = np.zeros(D, dtype=np.int64)
                        v[i], v[j], v[k], v[l] = si, sj, sk, sl
                        candidates.append(v)
    
    candidates = np.array(candidates)
    norms_sq = np.sum(candidates**2, axis=1)
    n_cand = len(candidates)
    
    # Try multiple orderings
    best_selected = None
    best_len = 0
    
    # Deterministic orderings
    orderings = [
        np.argsort(norms_sq),      # norm ascending
        np.argsort(-norms_sq),     # norm descending
    ]
    
    # Random orderings
    rng = np.random.RandomState(42)
    for _ in range(5):
        orderings.append(rng.permutation(n_cand))
    
    for order in orderings:
        if time.time() > deadline:
            break
        
        selected_idx = []
        max_norm_sq = 0
        
        for idx in order:
            if time.time() > deadline:
                break
            
            v = candidates[idx]
            v_norm_sq = norms_sq[idx]
            new_max = max(max_norm_sq, v_norm_sq)
            
            if selected_idx:
                selected_vecs = candidates[selected_idx]
                diffs = v - selected_vecs
                dist_sqs = np.sum(diffs**2, axis=1)
                if np.min(dist_sqs) < new_max:
                    continue
            
            selected_idx.append(idx)
            max_norm_sq = new_max
        
        if len(selected_idx) > best_len:
            best_len = len(selected_idx)
            best_selected = selected_idx[:]
    
    if best_selected is None or best_len == 0:
        # Fallback: weight-2 vectors only
        vectors = []
        for i, j in combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = np.zeros(D, dtype=np.int64)
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors)
    
    return candidates[best_selected]
# EVOLVE-BLOCK-END
