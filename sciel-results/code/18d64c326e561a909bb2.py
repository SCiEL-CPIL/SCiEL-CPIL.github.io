import numpy as np
import time

D = 11

def construct():
    start_time = time.time()
    deadline = 120.0
    
    # Generate all weight 4 integer vectors
    # Positions: combinations of 4 from 11
    # Signs: 16 patterns
    vectors = []
    import itertools
    for pos in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for p, s in zip(pos, signs):
                v[p] = s
            vectors.append(np.array(v, dtype=np.int64))
            
    vectors = np.array(vectors)
    n_vectors = len(vectors)
    
    # Precompute norms squared (all are 4)
    # We need min pairwise distance squared >= 4
    
    selected = []
    selected_vecs = []
    
    # Greedy selection
    # To make it faster, we can shuffle or just iterate.
    # 5280 vectors. O(N^2) check is ~27M ops. Fast in NumPy if vectorized?
    # But adding one by one requires checking against all selected.
    
    for i in range(n_vectors):
        if time.time() > deadline - 10:
            break
            
        v = vectors[i]
        if not selected_vecs:
            selected.append(i)
            selected_vecs.append(v)
            continue
            
        # Check distance to all selected
        # dist_sq = ||v - u||^2 = ||v||^2 + ||u||^2 - 2 v.u = 8 - 2 v.u
        # Need 8 - 2 v.u >= 4 => 2 v.u <= 4 => v.u <= 2
        
        # Calculate dot products with all selected
        # selected_vecs is list of arrays. Convert to matrix for batch dot?
        # If selected is large, this gets slow.
        
        # Optimization: if len(selected) > 100, use matrix multiplication
        if len(selected_vecs) > 50:
            mat = np.array(selected_vecs)
            dots = mat @ v
            if np.any(dots > 2):
                continue
        else:
            valid = True
            for u in selected_vecs:
                if np.dot(u, v) > 2:
                    valid = False
                    break
            if not valid:
                continue
                
        selected.append(i)
        selected_vecs.append(v)
        
    return np.array(selected_vecs)
