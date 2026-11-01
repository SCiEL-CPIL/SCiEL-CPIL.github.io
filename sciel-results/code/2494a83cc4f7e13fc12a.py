"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools


def construct():
    vectors = []
    target_k = 500
    
    # Generate vectors with squared norm 4 (norm 2)
    # Type 1: One coordinate is +/- 2, others 0
    for i in range(D):
        for s in (1, -1):
            v = [0] * D
            v[i] = 2 * s
            vectors.append(v)
            
    # Type 2: Four coordinates are +/- 1, others 0
    # We need target_k - len(vectors) more
    for indices in itertools.combinations(range(D), 4):
        for signs in itertools.product((1, -1), repeat=4):
            if len(vectors) >= target_k:
                break
            v = [0] * D
            for idx, s in zip(indices, signs):
                v[idx] = s
            vectors.append(v)
        if len(vectors) >= target_k:
            break
            
    return np.array(vectors, dtype=np.int64)
# EVOLVE-BLOCK-END
