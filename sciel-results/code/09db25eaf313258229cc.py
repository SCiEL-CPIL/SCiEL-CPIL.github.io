"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # 1. Generate candidate vectors with norm 2
    # Types:
    # A: +/- 2e_i (22 vectors)
    # B: +/- e_i +/- e_j +/- e_k +/- e_l (5280 vectors)
    
    candidates = []
    
    # Type A
    for i in range(D):
        for sign in (1, -1):
            v = [0] * D
            v[i] = sign * 2
            candidates.append(v)
            
    # Type B
    for indices in itertools.combinations(range(D), 4):
        i, j, k, l = indices
        for s1 in (1, -1):
            for s2 in (1, -1):
                for s3 in (1, -1):
                    for s4 in (1, -1):
                        v = [0] * D
                        v[i] = s1
                        v[j] = s2
                        v[k] = s3
                        v[l] = s4
                        candidates.append(v)
                        
    candidates = np.array(candidates, dtype=np.int64)
    n_cands = len(candidates)
    
    # 2. Precompute conflict graph using vectorized operations
    # Two vectors conflict if dist^2 < 4
    # We'll use a greedy approach with multiple random restarts
    
    def greedy_select(order):
        """Greedy selection given an ordering of candidates."""
        selected = []
        selected_indices = set()
        
        for idx in order:
            if time.time() > deadline:
                break
            v = candidates[idx]
            is_valid = True
            # Check against all already selected
            for sel_idx in selected_indices:
                diff = v - candidates[sel_idx]
                dist_sq = np.dot(diff, diff)
                if dist_sq < 4:
                    is_valid = False
                    break
            
            if is_valid:
                selected.append(idx)
                selected_indices.add(idx)
        
        return selected
    
    # Try multiple random restarts with different orderings
    best_selected = []
    
    # First, try a minimum-degree-like heuristic
    # Compute degree (number of conflicts) for each candidate
    # This is O(N^2) but we can vectorize
    
    # For speed, let's just do several random restarts
    for restart in range(20):
        if time.time() > deadline:
            break
        
        rng = random.Random(restart * 12345 + 67890)
        order = list(range(n_cands))
        rng.shuffle(order)
        
        selected = greedy_select(order)
        
        if len(selected) > len(best_selected):
            best_selected = selected
    
    if not best_selected:
        # Fallback: try one more with a different strategy
        order = list(range(n_cands))
        random.seed(42)
        random.shuffle(order)
        best_selected = greedy_select(order)
    
    if not best_selected:
        # Fallback to seed if something went wrong
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)

    result = candidates[best_selected]
    return result

# EVOLVE-BLOCK-END
