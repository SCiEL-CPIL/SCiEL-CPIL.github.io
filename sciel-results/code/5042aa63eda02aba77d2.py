import numpy as np
import time
import itertools

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin
    
    # Generate all weight-4 vectors with entries in {-1, 1}
    # Support: 4 indices from 0..10
    # Signs: 2^4 combinations
    candidates = []
    for support in itertools.combinations(range(D), 4):
        for signs in itertools.product([-1, 1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for idx, val in zip(support, signs):
                v[idx] = val
            candidates.append(v)
    
    candidates = np.array(candidates, dtype=np.int64)
    n_cand = len(candidates)
    
    best_vectors = None
    best_count = 0
    
    # Multiple restarts with different random orders
    rng = np.random.default_rng(42)
    
    for restart in range(100): # Max restarts, limited by time
        if time.time() > deadline - 5.0:
            break
            
        # Shuffle candidates
        order = rng.permutation(n_cand)
        shuffled_cands = candidates[order]
        
        selected = []
        selected_mat = np.empty((0, D), dtype=np.int64)
        
        for v in shuffled_cands:
            if time.time() > deadline - 5.0:
                break
            
            if len(selected) == 0:
                selected.append(v)
                selected_mat = v.reshape(1, D)
            else:
                # Check if v conflicts with any selected vector
                # Conflict if dot product >= 3
                dots = selected_mat @ v
                if np.any(dots >= 3):
                    continue
                else:
                    selected.append(v)
                    selected_mat = np.vstack([selected_mat, v.reshape(1, D)])
        
        if len(selected) > best_count:
            best_count = len(selected)
            best_vectors = np.array(selected, dtype=np.int64)
            
        # Early stop if we found a very good solution? 
        # Or if we are converging?
        # For now, just run all restarts within time limit.
        
    if best_vectors is None:
        # Fallback to seed if something went wrong
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)
        
    return best_vectors
