import numpy as np
import time
import random

D = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 110s safety margin
    
    # 1. Generate candidate pool: vectors with exactly 2 or 3 non-zero entries (each +/-1)
    candidates = []
    
    # m=2 vectors
    for i in range(D):
        for j in range(i+1, D):
            for si in (1, -1):
                for sj in (1, -1):
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    candidates.append(v)
                    
    # m=3 vectors
    for i in range(D):
        for j in range(i+1, D):
            for k in range(j+1, D):
                for si in (1, -1):
                    for sj in (1, -1):
                        for sk in (1, -1):
                            v = np.zeros(D, dtype=np.int64)
                            v[i] = si
                            v[j] = sj
                            v[k] = sk
                            candidates.append(v)
                            
    n_candidates = len(candidates)
    candidates = np.array(candidates) # (N, D)
    
    # 2. Precompute conflict graph
    # Conflict if dot(u, v) > 1
    # Dot products matrix
    dots = candidates @ candidates.T
    
    # Conflict mask: True if dot > 1
    conflict_mask = dots > 1
    # Diagonal is always True (dot with self is 2 or 3 > 1), which is correct (self-conflict)
    
    # Degrees
    degrees = np.sum(conflict_mask, axis=1)
    
    best_set_indices = []
    best_size = 0
    
    # Greedy heuristic with random restarts
    # We will try different orderings.
    # Heuristic 1: Random shuffle
    # Heuristic 2: Sort by degree + noise
    
    for restart in range(2000):
        if time.time() > deadline:
            break
            
        # Create an ordering
        # Mix of random and degree-based
        # If we sort by degree ascending, we pick "easy" ones first.
        # Let's add random noise to avoid getting stuck in local minima of the same order.
        noise = np.random.rand(n_candidates) * 10
        scores = degrees + noise
        order = np.argsort(scores)
        
        current_set = []
        # To speed up compatibility check, we can maintain a bitmask of selected nodes?
        # N=1540, bitmask fits in 25 ints or 1 numpy array of bools.
        # For each candidate in order, check if it conflicts with ANY selected.
        # conflict_mask[idx, current_set_indices] -> any()
        
        selected_bool = np.zeros(n_candidates, dtype=bool)
        
        for idx in order:
            # Check if idx conflicts with any selected
            # selected_bool is True for selected nodes.
            # conflict_mask[idx] is True for nodes that conflict with idx.
            # If any(selected_bool & conflict_mask[idx]) is True, then conflict.
            if not np.any(selected_bool & conflict_mask[idx]):
                current_set.append(idx)
                selected_bool[idx] = True
                
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set_indices = current_set
            
        # Early termination if we hit a high bound?
        # Kissing number is 554. We are using integer vectors.
        # If we get > 500, it's very good.
        
    return candidates[best_set_indices]
# EVOLVE-BLOCK-END
