"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    vectors = []
    
    # Generate all weight-2 vectors with entries +/-1
    # There are C(11, 2) * 4 = 55 * 4 = 220 such vectors.
    # These vectors have norm sqrt(2).
    # Minimum distance between any two distinct vectors in this set is 2.
    # 2 >= sqrt(2) is true.
    
    for i in range(D):
        for j in range(i + 1, D):
            for si in (1, -1):
                for sj in (1, -1):
                    if time.time() > deadline:
                        break
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    vectors.append(v)
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    if not vectors:
        return np.array([], dtype=np.int64).reshape(0, D)

    return np.array(vectors, dtype=np.int64)
# EVOLVE-BLOCK-END
