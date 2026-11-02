"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    vectors = []
    
    # Generate the 220 vectors of norm sqrt(2) in Z^11
    # These are vectors with exactly two non-zero entries, which are +/- 1.
    # This forms the minimal vectors of the D_11 lattice.
    # Condition: min pairwise distance >= max norm.
    # Norm = sqrt(2).
    # Min distance between distinct vectors of this type is sqrt(2).
    # So sqrt(2) >= sqrt(2) holds.
    
    for i in range(D):
        for j in range(i + 1, D):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i] = si
                    v[j] = sj
                    vectors.append(v)
                    
                    if time.time() > deadline:
                        break
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    if not vectors:
        # Fallback to unit vectors if something went wrong, though unlikely
        for i in range(D):
            v = [0] * D
            v[i] = 1
            vectors.append(v)
            v2 = [0] * D
            v2[i] = -1
            vectors.append(v2)

    return np.array(vectors, dtype=np.int64)
# EVOLVE-BLOCK-END
