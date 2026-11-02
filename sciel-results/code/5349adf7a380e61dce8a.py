"""Program for kissing number in dimension 11 using randomized greedy search."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools

def construct():
    # Type A vectors: 22 vectors, always included.
    type_a_vectors = []
    for i in range(D):
        for s in (1, -1):
            v = [0] * D
            v[i] = 2 * s
            type_a_vectors.append(v)
    type_a_arr = np.array(type_a_vectors, dtype=np.int64)
    
    # Type B vectors: 5280 vectors.
    type_b_vectors = []
    for support in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(support, signs):
                v[idx] = s
            type_b_vectors.append(v)
    type_b_arr = np.array(type_b_vectors, dtype=np.int64)
    
    n_b = len(type_b_arr)
    
    best_size = 0
    best_selected_indices = []
    
    # Number of random restarts
    # Tuned for speed: 2000 restarts
    n_restarts = 2000
    
    for _ in range(n_restarts):
        order = np.random.permutation(n_b)
        selected_vecs = []
        selected_indices = []
        
        for idx in order:
            v = type_b_arr[idx]
            
            if not selected_vecs:
                selected_vecs.append(v)
                selected_indices.append(idx)
                continue
            
            # Check compatibility
            compatible = True
            for sv in selected_vecs:
                if np.dot(sv, v) > 2:
                    compatible = False
                    break
            
            if compatible:
                selected_vecs.append(v)
                selected_indices.append(idx)
                
        if len(selected_vecs) > best_size:
            best_size = len(selected_vecs)
            best_selected_indices = selected_indices
            
    # Construct final result
    if best_selected_indices:
        best_b_arr = type_b_arr[best_selected_indices]
        result = np.vstack([type_a_arr, best_b_arr])
    else:
        result = type_a_arr
        
    return result
# EVOLVE-BLOCK-END
