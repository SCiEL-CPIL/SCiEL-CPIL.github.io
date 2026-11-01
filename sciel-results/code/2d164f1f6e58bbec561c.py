"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import itertools
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 110.0 # Safety margin
    
    vectors = []
    # Generate all vectors with exactly two non-zero entries of +/- 1
    # This corresponds to the minimal vectors of the D_11 lattice.
    # Norm squared is 2.
    # Min squared distance between distinct vectors is 2 (if supports overlap in 1 index) or 4 (if same support, opposite signs) or 2 (if disjoint? No, disjoint distance sq is 1+1+1+1=4? No.)
    
    # Let's re-verify disjoint distance.
    # v=(1,1,0..), w=(0,0,1,1..). v-w=(1,1,-1,-1..). Norm sq = 1+1+1+1=4.
    # Overlap 1: v=(1,1,0..), w=(1,0,1..). v-w=(0,1,-1..). Norm sq = 0+1+1=2.
    # Same support: v=(1,1,0..), w=(1,-1,0..). v-w=(0,2,0..). Norm sq = 4.
    # Same support: v=(1,1,0..), w=(-1,-1,0..). v-w=(2,2,0..). Norm sq = 8.
    
    # Max norm sq is 2.
    # Min dist sq is 2.
    # Condition: min_dist >= max_norm => sqrt(2) >= sqrt(2). True.
    
    for i, j in itertools.combinations(range(D), 2):
        if time.time() > deadline:
            break
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i] = si
                v[j] = sj
                vectors.append(v)
        
    if not vectors:
        # Fallback if something went wrong
        vectors = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
        
    return np.array(vectors, dtype=np.int64)
# EVOLVE-BLOCK-END
