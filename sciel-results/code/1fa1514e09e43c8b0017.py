"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import itertools
import numpy as np


def construct():
    vectors = []
    # Generate all weight-4 integer vectors
    # Indices: 0 to D-1
    for comb in itertools.combinations(range(D), 4):
        # Generate all 16 sign patterns for the 4 indices
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for i, s in zip(comb, signs):
                v[i] = s
            vectors.append(v)
    return np.array(vectors, dtype=np.int64)
# EVOLVE-BLOCK-END
