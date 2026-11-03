import numpy as np
import itertools
import random

D = 11

def construct():
    vectors = []
    for support in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(support, signs):
                v[idx] = s
            vectors.append(v)
    
    vectors = np.array(vectors, dtype=np.int64)
    
    # Shuffle indices
    indices = np.arange(len(vectors))
    np.random.shuffle(indices)
    vectors = vectors[indices]
    
    selected = []
    selected_matrix = None
    
    for v in vectors:
        if selected_matrix is None:
            selected_matrix = v.reshape(1, -1)
            selected.append(v)
        else:
            dots = selected_matrix @ v
            if not np.any(dots > 2):
                selected_matrix = np.vstack([selected_matrix, v.reshape(1, -1)])
                selected.append(v)
                
    return selected_matrix
