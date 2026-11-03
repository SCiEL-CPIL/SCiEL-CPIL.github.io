"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    vectors = []
    
    # Generate all vectors in Z^11 with exactly two non-zero entries, each +/- 1
    # These are the roots of the D_11 lattice
    for i in range(D):
        for j in range(i + 1, D):
            # Four sign combinations for (i, j)
            vectors.append([0]*D)
            vectors[-1][i] = 1
            vectors[-1][j] = 1
            
            vectors.append([0]*D)
            vectors[-1][i] = 1
            vectors[-1][j] = -1
            
            vectors.append([0]*D)
            vectors[-1][i] = -1
            vectors[-1][j] = 1
            
            vectors.append([0]*D)
            vectors[-1][i] = -1
            vectors[-1][j] = -1
            
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    return np.array(vectors, dtype=np.int64)
# EVOLVE-BLOCK-END
