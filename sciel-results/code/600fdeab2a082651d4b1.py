"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools

def construct():
    deadline = time.time() + 140.0  # safety margin before 150s kill
    
    # Generate candidate vectors of various types
    candidates = []
    
    # Type 1: D11 roots - vectors with two non-zero entries of ±1 (norm sqrt(2))
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i], v[j] = si, sj
                candidates.append(v)
    
    # Type 2: Vectors with three non-zero entries ±1 (norm sqrt(3))
    for i, j, k in itertools.combinations(range(D), 3):
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    v = [0] * D
                    v[i], v[j], v[k] = si, sj, sk
                    candidates.append(v)
    
    # Type 3: Vectors with four non-zero entries ±1 (norm sqrt(4) = 2)
    for i, j, k, l in itertools.combinations(range(D), 4):
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    for sl in (1, -1):
                        v = [0] * D
                        v[i], v[j], v[k], v[l] = si, sj, sk, sl
                        candidates.append(v)
    
    # Convert to numpy
    candidates = np.array(candidates, dtype=np.int64)
    
    # Compute squared norms
    norms_sq = np.sum(candidates ** 2, axis=1)
    
    # Sort by norm (smallest first) to prioritize adding low-norm vectors
    order = np.argsort(norms_sq, kind='stable')
    candidates_sorted = candidates[order]
    norms_sq_sorted = norms_sq[order]
    
    selected = []
    selected_norms_sq = []
    
    n_candidates = len(candidates_sorted)
    
    for idx in range(n_candidates):
        if time.time() > deadline:
            break
        
        v = candidates_sorted[idx]
        v_norm_sq = norms_sq_sorted[idx]
        
        # Check if v can be added: distance to all selected must be >= max(max_selected_norm, v_norm)
        if len(selected) == 0:
            selected.append(v)
            selected_norms_sq.append(v_norm_sq)
            continue
        
        # The threshold is max of all selected norms and this vector's norm
        max_norm_sq = max(max(selected_norms_sq), v_norm_sq)
        
        # Check pairwise distances
        # For efficiency, batch check
        if len(selected) <= 1000:
            # Direct check
            ok = True
            for sv in selected:
                diff = v - sv
                dist_sq = np.dot(diff, diff)
                if dist_sq < max_norm_sq:
                    ok = False
                    break
            if ok:
                selected.append(v)
                selected_norms_sq.append(v_norm_sq)
        else:
            # Use numpy for speed
            sel_arr = np.array(selected, dtype=np.int64)
            diffs = sel_arr - v
            dists_sq = np.sum(diffs ** 2, axis=1)
            if np.all(dists_sq >= max_norm_sq):
                selected.append(v)
                selected_norms_sq.append(v_norm_sq)
    
    if not selected:
        # Fallback to D11 roots
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)
    
    return np.array(selected, dtype=np.int64)

# EVOLVE-BLOCK-END
