import numpy as np
import itertools

D = 11

def construct():
    """
    Constructs the set of 220 minimal vectors (roots) of the D_11 lattice.
    These are all integer vectors in R^11 with exactly two non-zero entries,
    each being +1 or -1.
    
    Properties:
    - All vectors have norm sqrt(2).
    - The minimum pairwise distance between any two distinct vectors is sqrt(2).
    - Condition: min_dist >= max_norm => sqrt(2) >= sqrt(2) holds.
    - Size: 2 * C(11, 2) = 220.
    
    This is believed to be the maximum size for integer vectors in R^11
    satisfying the condition, as adding vectors of higher norm (e.g., sqrt(3))
    requires removing multiple existing vectors due to distance constraints.
    """
    vectors = []
    # Generate all pairs of indices
    for i, j in itertools.combinations(range(D), 2):
        # Generate all sign combinations (+1, +1), (+1, -1), (-1, +1), (-1, -1)
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i] = si
                v[j] = sj
                vectors.append(v)
    
    return np.array(vectors, dtype=np.int64)
