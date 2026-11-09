"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    def _check_validity(vectors):
        """Check if the set of vectors satisfies the constraint."""
        if len(vectors) == 0:
            return True
        norms_sq = np.sum(vectors * vectors, axis=1)
        max_norm_sq = np.max(norms_sq)
        n = len(vectors)
        # Check all pairwise distances
        for i in range(n):
            for j in range(i+1, n):
                diff = vectors[i] - vectors[j]
                dist_sq = np.sum(diff * diff)
                if dist_sq < max_norm_sq - 1e-9:
                    return False
        return True
    
    # Generate base vectors: all pairs with +/-1
    base_vectors = []
    for i in range(D):
        for j in range(i+1, D):
            for si in (1, -1):
                for sj in (1, -1):
                    v = np.zeros(D, dtype=np.int64)
                    v[i], v[j] = si, sj
                    base_vectors.append(v)
    
    base_vectors = np.array(base_vectors, dtype=np.int64)
    
    # Also add single-component vectors
    single_vectors = []
    for i in range(D):
        for val in (1, -1):
            v = np.zeros(D, dtype=np.int64)
            v[i] = val
            single_vectors.append(v)
    
    all_candidates = np.vstack([base_vectors, single_vectors])
    
    # Remove duplicates
    all_candidates = np.unique(all_candidates, axis=0)
    
    # Now use a greedy approach with vectorized distance checks
    # Sort candidates by norm (prefer higher norm vectors as they are more "spread")
    norms_sq = np.sum(all_candidates * all_candidates, axis=1)
    sort_idx = np.argsort(-norms_sq)
    sorted_candidates = all_candidates[sort_idx]
    sorted_norms_sq = norms_sq[sort_idx]
    
    selected = []
    selected_norms_sq = []
    max_norm_sq = 0
    
    n_candidates = len(sorted_candidates)
    
    for idx in range(n_candidates):
        if time.time() > deadline - 5:
            break
        
        v = sorted_candidates[idx]
        v_norm_sq = sorted_norms_sq[idx]
        new_max_norm_sq = max(max_norm_sq, v_norm_sq)
        
        # Check if v can be added: dist to all selected >= new_max_norm
        if len(selected) == 0:
            selected.append(v)
            selected_norms_sq.append(v_norm_sq)
            max_norm_sq = new_max_norm_sq
            continue
        
        # Vectorized distance check
        selected_arr = np.array(selected)
        diffs = v - selected_arr
        dists_sq = np.sum(diffs * diffs, axis=1)
        
        if np.all(dists_sq >= new_max_norm_sq - 1e-9):
            selected.append(v)
            selected_norms_sq.append(v_norm_sq)
            max_norm_sq = new_max_norm_sq
    
    # Try a few random restarts with different orderings
    best_count = len(selected)
    best_selected = selected[:]
    
    for restart in range(5):
        if time.time() > deadline - 3:
            break
        
        # Random permutation
        perm = np.random.permutation(n_candidates)
        shuffled_candidates = sorted_candidates[perm]
        shuffled_norms_sq = sorted_norms_sq[perm]
        
        selected_r = []
        max_norm_sq_r = 0
        
        for idx in range(n_candidates):
            if time.time() > deadline - 2:
                break
            
            v = shuffled_candidates[idx]
            v_norm_sq = shuffled_norms_sq[idx]
            new_max_norm_sq = max(max_norm_sq_r, v_norm_sq)
            
            if len(selected_r) == 0:
                selected_r.append(v)
                max_norm_sq_r = new_max_norm_sq
                continue
            
            selected_arr = np.array(selected_r)
            diffs = v - selected_arr
            dists_sq = np.sum(diffs * diffs, axis=1)
            
            if np.all(dists_sq >= new_max_norm_sq - 1e-9):
                selected_r.append(v)
                max_norm_sq_r = new_max_norm_sq
        
        if len(selected_r) > best_count:
            best_count = len(selected_r)
            best_selected = selected_r[:]
    
    result = np.array(best_selected, dtype=np.int64)
    
    # Verify validity
    if not _check_validity(result):
        # Fall back to base vectors only
        return base_vectors
    
    return result

# EVOLVE-BLOCK-END
