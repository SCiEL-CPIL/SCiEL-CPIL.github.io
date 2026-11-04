"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import itertools
import numpy as np

D = 11

# EVOLVE-BLOCK-START

def construct():
    vectors = []
    # Generate all vectors with exactly two non-zero entries (+/- 1)
    # These have norm sqrt(2).
    # Minimum distance between any two such vectors is sqrt(2).
    # Condition: min_dist >= max_norm => sqrt(2) >= sqrt(2) holds.
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i] = si
                v[j] = sj
                vectors.append(v)
    
    return np.array(vectors, dtype=np.int64)

# EVOLVE-BLOCK-END
