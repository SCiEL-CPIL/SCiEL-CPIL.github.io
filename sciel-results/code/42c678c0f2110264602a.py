"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import itertools
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Generate weight 4 vectors
    vectors = []
    for idxs in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(idxs, signs):
                v[idx] = s
            vectors.append(v)
            
    V = np.array(vectors, dtype=np.int64)
    n = len(V)
    
    # Compute Gram matrix
    G = V @ V.T
    
    best_set = []
    best_size = 0
    
    # Greedy with random restarts
    num_restarts = 1000
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        candidates = np.arange(n)
        selected_indices = []
        
        while len(candidates) > 0:
            # Pick a random candidate
            idx_pos = np.random.randint(len(candidates))
            v_idx = candidates[idx_pos]
            
            selected_indices.append(v_idx)
            
            # Filter candidates: keep those with dot <= 2 with v_idx
            dots = G[v_idx, candidates]
            mask = dots <= 2
            candidates = candidates[mask]
            
            if time.time() > deadline:
                if len(selected_indices) > best_size:
                    best_set = selected_indices
                    best_size = len(selected_indices)
                return np.array([V[i] for i in best_set], dtype=np.int64)
        
        if len(selected_indices) > best_size:
            best_set = selected_indices
            best_size = len(selected_indices)
            
    # Fallback if no time or failure
    if not best_set:
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)
        
    return np.array([V[i] for i in best_set], dtype=np.int64)
# EVOLVE-BLOCK-END
