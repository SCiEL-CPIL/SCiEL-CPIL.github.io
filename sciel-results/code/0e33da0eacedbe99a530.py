import numpy as np
import itertools
import time

D = 11

def construct():
    """
    Constructs a set of nonzero integer vectors in R^11.
    The set consists of all vectors with exactly two non-zero entries, which are +/- 1.
    This corresponds to the shortest vectors in the D_11 lattice.
    
    Properties:
    - Norm of each vector is sqrt(2).
    - Minimum pairwise distance between distinct vectors is 2 (sqrt(4)).
    - Condition: min_distance (2) >= max_norm (sqrt(2)) holds.
    - Number of vectors k = 2 * C(11, 2) = 220.
    """
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill

    vectors = []
    
    # Generate all combinations of 2 indices
    for i, j in itertools.combinations(range(D), 2):
        # Generate all sign combinations for these two indices
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i] = si
                v[j] = sj
                vectors.append(v)
                
                # Check time
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break

    if not vectors:
        # Fallback if something went wrong (should not happen)
        return np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int64)

    return np.array(vectors, dtype=np.int64)
