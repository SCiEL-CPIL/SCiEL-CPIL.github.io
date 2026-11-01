import numpy as np
import itertools

def construct():
    D = 11
    
    # List to store vectors
    vectors = []
    
    # Type 1: One entry is +/- 2, rest are 0.
    # Norm squared = 4. Sum = +/- 2 (even).
    # Count: 11 positions * 2 signs = 22
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            vectors.append(v)
            
    # Type 2: Four entries are +/- 1, rest are 0.
    # Norm squared = 4. Sum is even.
    # Count: C(11, 4) * 2^4 = 330 * 16 = 5280
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            vectors.append(v)
            
    # Convert to numpy array
    result = np.array(vectors, dtype=np.int64)
    
    return result
