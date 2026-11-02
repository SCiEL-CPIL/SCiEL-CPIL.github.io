"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    # Generate all weight-4 vectors
    # These are mutually compatible (dot <= 2)
    vectors = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            vectors.append(v)
            
    result = np.array(vectors, dtype=np.int64)
    
    return result

# EVOLVE-BLOCK-END
