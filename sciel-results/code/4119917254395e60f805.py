"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s limit
    
    # Generate weight-4 vectors with symmetry breaking:
    # For each set of 4 positions, only include sign patterns where the first
    # non-zero coordinate is positive. This reduces 16 sign combos to 8.
    # C(11,4) = 330 positions, 8 sign combos -> 2640 vectors
    candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            # Symmetry breaking: first non-zero coordinate must be positive
            if signs[0] == -1:
                continue
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates.append(v)
    
    candidates = np.array(candidates, dtype=np.int64)
    n_candidates = len(candidates)
    
    # Norm squared of each candidate = 4
    # Constraint: |u-v|^2 >= 4 for all pairs, i.e., u·v <= 2
    
    # Greedy min-degree selection with multiple random orderings
    best_selected = []
    best_count = 0
    
    np.random.seed(42)
    for trial in range(500):
        if time.time() > deadline - 10.0:
            break
        
        # Random permutation of candidates
        perm = np.random.permutation(n_candidates)
        
        selected = []
        selected_idx = set()
        
        for idx in perm:
            if time.time() > deadline - 5.0:
                break
            
            v = candidates[idx]
            if not selected:
                selected.append(v)
                selected_idx.add(idx)
                continue
            
            # Check constraint against all selected
            dots = np.dot(selected, v)
            if np.all(dots <= 2):
                selected.append(v)
                selected_idx.add(idx)
        
        if len(selected) > best_count:
            best_count = len(selected)
            best_selected = selected[:]
        
        # Early stop if we've found a very good solution
        if best_count >= 1500:
            break
    
    if not best_selected:
        # Fallback: return a minimal valid set
        v = [0] * D
        v[0] = 1
        v[1] = 1
        return np.array([v], dtype=np.int64)
    
    return np.array(best_selected, dtype=np.int64)
# EVOLVE-BLOCK-END
