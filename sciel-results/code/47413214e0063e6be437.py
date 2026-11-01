"""Kissing number in dimension 11 using E8 ⊕ A3 lattice construction."""
import numpy as np
import time
from itertools import product

D = 11

def construct():
    start_time = time.time()
    
    # E8 root lattice vectors: all permutations of (±1, ±1, 0, 0, 0, 0, 0, 0) 
    # and (±1/2, ±1/2, ..., ±1/2) with even number of minus signs
    # We'll use integer vectors by scaling: use vectors with entries in {-1, 0, 1}
    # norm^2 = 4 (i.e., weight 4 with ±1 entries)
    
    # Actually, let's use the standard E8 construction with integer vectors:
    # Type 1: (±1, ±1, 0, 0, 0, 0, 0, 0) and permutations: C(8,2)*4 = 28*4 = 112
    # Type 2: (±1/2,...,±1/2) scaled by 2: (±1,...,±1) with even number of 1s: 2^7 = 128
    # Total E8: 240 vectors
    
    e8_vectors = []
    
    # Type 1: two ±1's, rest 0
    for i in range(8):
        for j in range(i+1, 8):
            for si in [1, -1]:
                for sj in [1, -1]:
                    v = [0] * 8
                    v[i] = si
                    v[j] = sj
                    e8_vectors.append(v)
    
    # Type 2: all ±1 with even number of -1s (or equivalently, product of signs = +1)
    for signs in product([1, -1], repeat=8):
        if sum(1 for s in signs if s == -1) % 2 == 0:
            e8_vectors.append(list(signs))
    
    # A3 = D3 lattice vectors: (±1, ±1, 0) and permutations: C(3,2)*4 = 3*4 = 12
    a3_vectors = []
    for i in range(3):
        for j in range(i+1, 3):
            for si in [1, -1]:
                for sj in [1, -1]:
                    v = [0] * 3
                    v[i] = si
                    v[j] = sj
                    a3_vectors.append(v)
    
    # Combine: E8 in first 8 coords, A3 in last 3 coords
    all_vectors = []
    for e8v in e8_vectors:
        # A3 is zero in last 3 coords
        v = e8v + [0, 0, 0]
        all_vectors.append(v)
    for a3v in a3_vectors:
        # E8 is zero in first 8 coords
        v = [0] * 8 + a3v
        all_vectors.append(v)
    
    result = np.array(all_vectors, dtype=np.int64)
    
    # Verify: all nonzero, min pairwise distance >= max norm
    norms_sq = np.sum(result ** 2, axis=1)
    max_norm_sq = np.max(norms_sq)
    
    # Check pairwise distances
    diffs = result[:, np.newaxis, :] - result[np.newaxis, :, :]
    dist_sq = np.sum(diffs ** 2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    min_dist_sq = np.min(dist_sq)
    
    # The constraint: min_dist >= max_norm, i.e., min_dist_sq >= max_norm_sq
    # Here max_norm_sq = 4, min_dist_sq should be >= 4
    
    return result

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
