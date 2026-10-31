import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10 second margin before 150s limit
    
    # Generate all weight-4 vectors
    candidates = []
    for combo in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(combo, signs):
                v[idx] = s
            candidates.append(v)
    
    candidates = np.array(candidates, dtype=np.int8)
    n_candidates = len(candidates)
    
    # Precompute dot products and conflict mask
    # Conflict if dot product >= 3 (since dist^2 >= 4 requires dot <= 2)
    dots = candidates @ candidates.T
    conflict_mask = dots >= 3
    
    best_size = 0
    best_indices = []
    
    # Randomized greedy restarts
    # Each restart is O(N * degree) which is very fast
    max_restarts = 20000
    
    for r in range(max_restarts):
        if time.time() > deadline:
            break
        
        indices = np.random.permutation(n_candidates)
        selected = []
        blocked = np.zeros(n_candidates, dtype=bool)
        
        for i in indices:
            if not blocked[i]:
                selected.append(i)
                # Block i and all its neighbors
                blocked[i] = True
                # Get indices of conflicts
                # Using np.where is efficient enough for N=5280
                conflicts = np.where(conflict_mask[i])[0]
                blocked[conflicts] = True
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_indices = selected
    
    if not best_indices:
        return np.empty((0, D), dtype=np.int64)
    
    return candidates[best_indices].astype(np.int64)

# EVOLVE-BLOCK-END
