import numpy as np
import itertools

def construct():
    D = 11
    vectors = []
    
    # Generate all vectors with norm squared 2
    # These are weight-2 vectors with entries +/-1
    for pos1 in range(D):
        for pos2 in range(pos1 + 1, D):
            for s1 in [1, -1]:
                for s2 in [1, -1]:
                    v = [0] * D
                    v[pos1] = s1
                    v[pos2] = s2
                    vectors.append(v)
                    
    # Generate all vectors with norm squared 4
    # Type 1: Weight 2 with entries +/-2
    for pos1 in range(D):
        for pos2 in range(pos1 + 1, D):
            for s1 in [2, -2]:
                for s2 in [2, -2]:
                    v = [0] * D
                    v[pos1] = s1
                    v[pos2] = s2
                    vectors.append(v)
                    
    # Type 2: Weight 4 with entries +/-1
    for support in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, pos in enumerate(support):
                v[pos] = signs[idx]
            vectors.append(v)
            
    return np.array(vectors, dtype=np.int64)
