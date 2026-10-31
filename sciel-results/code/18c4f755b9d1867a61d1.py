import numpy as np
import itertools

def construct():
    D = 11
    vectors = []
    
    # Part 1: Scaled D11 minimal vectors
    # Two non-zeros +/- 2 in any 2 of 11 dimensions
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i] = 2 * si
                v[j] = 2 * sj
                vectors.append(v)
                
    # Part 2: E8 Type 2 vectors (embedded in first 8 dims)
    # Eight non-zeros +/- 1 in first 8 dims, even number of -1s
    for mask in range(256):
        v = [0] * D
        minus_count = 0
        for k in range(8):
            if (mask >> k) & 1:
                v[k] = -1
                minus_count += 1
            else:
                v[k] = 1
        if minus_count % 2 == 0:
            vectors.append(v)
            
    # Convert to numpy array for fast operations
    vec_array = np.array(vectors, dtype=np.int64)
    max_norm_sq = np.max(np.sum(vec_array**2, axis=1))
    
    # Candidate pool: Vectors with 2 non-zeros +/- 1 in first 8 dims
    # These have norm sqrt(2).
    # Check if they can be added.
    # Condition: |v_new - v_old|^2 >= max_norm_sq
    # |v_new|^2 + |v_old|^2 - 2 v_new . v_old >= max_norm_sq
    # 2 + 8 - 2 v_new . v_old >= 8  (assuming v_old is norm sqrt(8))
    # 10 - 2 dot >= 8 => dot <= 1
    # If v_old is Part 1 (norm sqrt(8)), same condition.
    
    candidates = []
    for i, j in itertools.combinations(range(8), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i] = si
                v[j] = sj
                candidates.append(v)
                
    cand_array = np.array(candidates, dtype=np.int64)
    
    new_vectors = []
    for c in cand_array:
        c_sq = np.sum(c**2)
        # Check against all existing vectors
        # We need |c - v|^2 >= max_norm_sq for all v in vec_array
        # |c|^2 + |v|^2 - 2 c.v >= max_norm_sq
        # c_sq + |v|^2 - 2 c.v >= max_norm_sq
        # 2 c.v <= c_sq + |v|^2 - max_norm_sq
        # We can compute max dot product with existing vectors.
        
        # Efficient check:
        # c.v for all v in vec_array
        dots = vec_array @ c
        norms_sq = np.sum(vec_array**2, axis=1)
        
        # Condition: c_sq + norms_sq - 2*dots >= max_norm_sq
        # 2*dots <= c_sq + norms_sq - max_norm_sq
        lhs = 2 * dots
        rhs = c_sq + norms_sq - max_norm_sq
        if np.all(lhs <= rhs):
            new_vectors.append(c)
            
    if new_vectors:
        vec_array = np.vstack([vec_array, np.array(new_vectors, dtype=np.int64)])
        
    return vec_array
