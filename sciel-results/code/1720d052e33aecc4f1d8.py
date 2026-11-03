import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Strategy 1: D_11 lattice minimal vectors (220 vectors)
    # These are permutations of (1, 1, 0, ..., 0) with all sign combinations
    d11_vectors = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i] = si
                v[j] = sj
                d11_vectors.append(v)
    
    d11_array = np.array(d11_vectors, dtype=np.int64)
    
    # Strategy 2: E_8 x D_3 construction (240 + 12 = 252 vectors)
    # E_8 minimal vectors scaled by 2 to be integers
    # Type 1: (±2, ±2, 0, ..., 0)
    # Type 2: (±1, ±1, ..., ±1)
    
    e8_vectors = []
    # Type 1
    for i, j in itertools.combinations(range(8), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * 8
                v[i] = 2 * si
                v[j] = 2 * sj
                e8_vectors.append(v)
                
    # Type 2
    # All combinations of ±1
    for signs in itertools.product([1, -1], repeat=8):
        e8_vectors.append(list(signs))
        
    e8_array = np.array(e8_vectors, dtype=np.int64)
    
    # D_3 minimal vectors (12 vectors)
    # Permutations of (1, 1, 0) with signs
    d3_vectors = []
    for i, j in itertools.combinations(range(3), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * 3
                v[i] = si
                v[j] = sj
                d3_vectors.append(v)
                
    d3_array = np.array(d3_vectors, dtype=np.int64)
    
    # Combine E_8 and D_3 into R^11
    # E_8 part is in first 8 dims, D_3 in last 3
    combined_vectors = []
    
    # Add E_8 vectors (padded with 0s)
    for v in e8_array:
        combined_vectors.append(np.concatenate([v, np.zeros(3, dtype=np.int64)]))
        
    # Add D_3 vectors (padded with 0s at start)
    for v in d3_array:
        combined_vectors.append(np.concatenate([np.zeros(8, dtype=np.int64), v]))
        
    combined_array = np.array(combined_vectors, dtype=np.int64)
    
    def is_valid(arr):
        if len(arr) == 0:
            return True
        norms = np.sqrt(np.sum(arr**2, axis=1))
        max_norm = np.max(norms)
        # Compute pairwise distances
        # Use broadcasting for small N, but 252 is small enough
        # dist^2 = ||u||^2 + ||v||^2 - 2 u.v
        dots = arr @ arr.T
        norms_sq = np.sum(arr**2, axis=1)
        dist_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * dots
        np.fill_diagonal(dist_sq, np.inf)
        min_dist_sq = np.min(dist_sq)
        return min_dist_sq >= max_norm**2 - 1e-9
        
    best_result = None
    best_score = 0
    
    # Check D_11
    if is_valid(d11_array):
        if len(d11_array) > best_score:
            best_score = len(d11_array)
            best_result = d11_array
            
    # Check Combined
    if time.time() < deadline:
        if is_valid(combined_array):
            if len(combined_array) > best_score:
                best_score = len(combined_array)
                best_result = combined_array
                
    if best_result is None:
        return d11_array
    return best_result

