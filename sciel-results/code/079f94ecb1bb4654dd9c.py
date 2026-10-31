import numpy as np
import itertools

def construct():
    D = 11
    
    # Part 1: Norm squared 2 vectors (D11 minimal vectors)
    # These are vectors with two non-zero entries of +/-1
    vectors = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i] = si
                v[j] = sj
                vectors.append(v)
                
    vec_array = np.array(vectors, dtype=np.int64)
    
    # Part 2: Generate candidates with norm squared 4
    candidates = []
    
    # Type A: (±2, 0...)
    for i in range(D):
        for s in (1, -1):
            v = [0] * D
            v[i] = 2 * s
            candidates.append(v)
            
    # Type B: (±2, ±2, 0...)
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i] = 2 * si
                v[j] = 2 * sj
                candidates.append(v)
                
    # Type C: (±1, ±1, ±1, ±1, 0...)
    for indices in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for k, idx in enumerate(indices):
                v[idx] = signs[k]
            candidates.append(v)
            
    cand_array = np.array(candidates, dtype=np.int64)
    
    best_count = len(vec_array)
    best_vec_list = vec_array.copy()
    
    # Greedy addition with random restarts
    # We use a fixed seed for reproducibility in the evaluation environment if needed,
    # but random is better for finding good configurations.
    np.random.seed(42)
    
    for _ in range(20):
        current = vec_array.copy()
        current_norms_sq = np.sum(current**2, axis=1)
        
        # Shuffle indices of candidates
        indices = np.arange(len(cand_array))
        np.random.shuffle(indices)
        shuffled_cands = cand_array[indices]
        
        for u in shuffled_cands:
            # Compute dot products of u with all current vectors
            dots = current @ u
            # Condition: 2 * dot <= norm_v_sq
            valid = np.all(2 * dots <= current_norms_sq)
            
            if valid:
                current = np.vstack([current, u])
                current_norms_sq = np.append(current_norms_sq, 4)
                
        if len(current) > best_count:
            best_count = len(current)
            best_vec_list = current
            
    return best_vec_list
