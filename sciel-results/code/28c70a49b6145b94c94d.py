"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import itertools
import time

D = 11

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a set of integer vectors in R^11 such that the minimum pairwise
    distance is at least the maximum vector norm.
    
    The strategy uses all vectors of the form +/- e_i +/- e_j (i != j), which have
    norm sqrt(2). The minimum distance between any two distinct such vectors is
    sqrt(2) (if they share one index) or larger. Thus, the condition is satisfied.
    Total vectors: 4 * C(11, 2) = 220.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    vectors = []
    
    # Generate all vectors with exactly two non-zero entries, each +/-1
    # These have norm sqrt(2).
    for i in range(D):
        for j in range(i + 1, D):
            # 4 combinations of signs for i and j
            for si in (1, -1):
                for sj in (1, -1):
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    vectors.append(v)
                    
                    if time.time() > deadline:
                        # Return what we have if we hit the deadline (unlikely for 220 items)
                        return np.array(vectors, dtype=np.int64)
                        
    return np.array(vectors, dtype=np.int64)
# EVOLVE-BLOCK-END
