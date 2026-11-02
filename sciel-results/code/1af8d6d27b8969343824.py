import numpy as np
import itertools
import time
import random

D = 11

def construct():
    start_time = time.time()
    deadline = 140.0
    
    # 1. Generate N=2 vectors
    v2_list = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i], v[j] = si, sj
                v2_list.append(tuple(v))
                
    # 2. Generate supports for N=4 Type B
    supports = list(itertools.combinations(range(D), 4))
    n_sup = len(supports)
    
    # Precompute conflict matrix for supports
    # Conflict if intersection > 2
    conflict = np.zeros((n_sup, n_sup), dtype=bool)
    sets = [set(s) for s in supports]
    for i in range(n_sup):
        for j in range(i+1, n_sup):
            if len(sets[i] & sets[j]) > 2:
                conflict[i][j] = True
                conflict[j][i] = True
                
    # Greedy search for Max Independent Set in this graph
    best_size = 0
    best_mask = np.zeros(n_sup, dtype=bool)
    
    num_restarts = 200
    for _ in range(num_restarts):
        if time.time() > deadline:
            break
            
        order = np.random.permutation(n_sup)
        used = np.zeros(n_sup, dtype=bool)
        current_mask = np.zeros(n_sup, dtype=bool)
        
        for idx in order:
            if used[idx]:
                continue
            # Check if any neighbor is used?
            # Actually, we mark neighbors as used when we pick a vertex.
            # So if we pick idx, we mark idx and all neighbors as used.
            # But we need to check if idx is available (not used).
            # The logic above:
            # If we pick idx, we set used[idx]=True and used[neighbors]=True.
            # So we just need to check if used[idx] is False.
            
            # Wait, the standard greedy for MIS:
            # Iterate vertices. If not used, add to set, mark self and neighbors used.
            
            current_mask[idx] = True
            used[idx] = True
            # Mark neighbors
            # Find neighbors of idx
            # conflict[idx] is a boolean vector
            used[conflict[idx]] = True
            
        if np.sum(current_mask) > best_size:
            best_size = np.sum(current_mask)
            best_mask = current_mask.copy()
            
    # Construct final vectors
    final_vectors = list(v2_list)
    
    # Add 11 Type A vectors
    for i in range(D):
        v = [0] * D
        v[i] = 2
        final_vectors.append(tuple(v))
        
    # Add Type B vectors for selected supports
    for i in range(n_sup):
        if best_mask[i]:
            s = supports[i]
            idxs = list(s)
            for signs in itertools.product([1, -1], repeat=4):
                v = [0] * D
                for k, idx in enumerate(idxs):
                    v[idx] = signs[k]
                final_vectors.append(tuple(v))
                
    return np.array(final_vectors, dtype=np.int64)
