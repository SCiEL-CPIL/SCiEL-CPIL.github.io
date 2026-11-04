"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    vectors = []
    
    # 1. Generate E8 lattice vectors (scaled by 2) in first 8 dimensions
    # Type 1: Permutations of (±2, ±2, 0, ..., 0)
    for i, j in itertools.combinations(range(8), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i] = 2 * si
                v[j] = 2 * sj
                vectors.append(v)
                
    # Type 2: Permutations of (±1, ..., ±1) with even number of minus signs
    # There are 2^8 = 256 such vectors, half have even number of minus signs (128)
    for bits in range(256):
        # Check parity of set bits (minus signs)
        # Represent minus signs as 1s in the bitmask
        # We need even number of minus signs
        # Let's generate all 256 and filter
        v = [0] * D
        minus_count = 0
        for k in range(8):
            if (bits >> k) & 1:
                v[k] = -1
                minus_count += 1
            else:
                v[k] = 1
        if minus_count % 2 == 0:
            vectors.append(v)
            
    if time.time() > deadline:
        return np.array(vectors, dtype=np.int64)

    # 2. Generate D3 lattice vectors (scaled by 2) in last 3 dimensions (indices 8, 9, 10)
    # Type: Permutations of (±2, ±2, 0) in the last 3 positions
    # Indices 8, 9, 10
    for i, j in itertools.combinations([8, 9, 10], 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i] = 2 * si
                v[j] = 2 * sj
                vectors.append(v)
                
    if time.time() > deadline:
        return np.array(vectors, dtype=np.int64)

    return np.array(vectors, dtype=np.int64)
# EVOLVE-BLOCK-END
