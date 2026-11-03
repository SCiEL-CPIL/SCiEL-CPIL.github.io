"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools

def construct():
    vectors = []
    
    # Generate all weight-2 vectors with entries +/-1
    # There are C(11, 2) = 55 supports
    # Each support has 4 sign combinations
    for i in range(D):
        for j in range(i + 1, D):
            # Support {i, j}
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i] = si
                    v[j] = sj
                    vectors.append(v)
    
    return np.array(vectors, dtype=np.int64)
# EVOLVE-BLOCK-END
