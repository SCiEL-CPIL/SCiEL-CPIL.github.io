import numpy as np
import itertools

D = 11

def construct():
    # Generate all weight-2 vectors in {-1, 0, 1}^11
    # Norm^2 = 2
    # Condition for equal norm R^2=2: dot <= 1
    # Distinct weight-2 vectors have dot in {-2, 0, 1}
    # So all are mutually compatible.
    
    vectors = []
    for positions in itertools.combinations(range(D), 2):
        for signs in itertools.product([1, -1], repeat=2):
            v = [0] * D
            v[positions[0]] = signs[0]
            v[positions[1]] = signs[1]
            vectors.append(v)
            
    return np.array(vectors, dtype=np.int64)
