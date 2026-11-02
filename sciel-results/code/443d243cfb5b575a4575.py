import numpy as np
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115

    D = 11
    
    # Generate all integer vectors with norm squared 4
    # Type 1: 2 * e_i
    # Type 2: +/- e_i +/- e_j +/- e_k +/- e_l
    
    vectors = []
    
    # Type 1: 2 * e_i
    for i in range(D):
        v = np.zeros(D, dtype=np.int64)
        v[i] = 2
        vectors.append(v)
        v[i] = -2
        vectors.append(v)
        
    # Type 2: four 1s
    for support in itertools.combinations(range(D), 4):
        for signs in itertools.product([-1, 1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for pos, sign in zip(support, signs):
                v[pos] = sign
            vectors.append(v)
            
    vecs = np.array(vectors, dtype=np.float64)
    n_vecs = len(vecs)
    
    # We want to find a large subset with pairwise dot product <= 2
    # Greedy with random restarts
    
    best_set = []
    best_count = 0
    
    # Precompute nothing? Just loop.
    # n_vecs is ~5300.
    # K is expected to be ~250.
    # 5300 * 250 = 1.3M dots.
    
    max_restarts = 50
    for r in range(max_restarts):
        if time.time() > deadline - 10:
            break
            
        rng = np.random.RandomState(r)
        indices = rng.permutation(n_vecs)
        
        selected_indices = []
        selected_vectors = []
        
        for idx in indices:
            if time.time() > deadline - 5:
                break
                
            v = vecs[idx]
            
            if len(selected_vectors) > 0:
                # Check compatibility
                # Convert selected to array for batch dot?
                # selected_vectors is a list.
                # Let's keep an array of selected vectors for efficient dot.
                # But we add one by one.
                # We can maintain a matrix of selected vectors.
                
                # For speed, if selected is small, loop.
                # If selected is large, use matrix.
                
                compatible = True
                for sv in selected_vectors:
                    if np.dot(v, sv) > 2:
                        compatible = False
                        break
                        
                if not compatible:
                    continue
            
            selected_indices.append(idx)
            selected_vectors.append(v)
            
        if len(selected_indices) > best_count:
            best_count = len(selected_indices)
            best_set = selected_indices[:]
            
    # Convert best_set to array
    if best_count > 0:
        result = vecs[best_set].astype(np.int64)
    else:
        # Fallback: just return roots?
        # Generate roots
        roots = []
        for i in range(D):
            for j in range(i + 1, D):
                for si in [-1, 1]:
                    for sj in [-1, 1]:
                        v = np.zeros(D, dtype=np.int64)
                        v[i] = si
                        v[j] = sj
                        roots.append(v)
        result = np.array(roots, dtype=np.int64)
        
    return result
