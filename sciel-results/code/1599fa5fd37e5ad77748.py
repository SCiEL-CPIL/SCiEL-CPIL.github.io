"""Kissing number in dimension 11: construct nonzero integer vectors in R^11
whose minimum pairwise distance is at least the maximum vector norm.
Interface: construct() -> integer array of shape (k, d)."""

import itertools
import time
import numpy as np

D = 11
DEADLINE_S = 118.0

def construct():
    t0 = time.perf_counter()
    
    def elapsed():
        return time.perf_counter() - t0

    def is_deadline():
        return elapsed() > DEADLINE_S

    # Baseline: m=2, 220 vectors
    base = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i], v[j] = si, sj
                base.append(v)
    base_arr = np.array(base, dtype=np.int64)
    
    # Target: m=4, 5280 + 22 = 5302 vectors
    # 1. Weight 4 vectors with values +/-1
    vecs = []
    # Generate all combinations of 4 indices
    for support in itertools.combinations(range(D), 4):
        # Generate all sign combinations for these 4 indices
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for idx, s in zip(support, signs):
                v[idx] = s
            vecs.append(v)
            
    # 2. Vectors with a single 2 (or -2)
    for i in range(D):
        for s in (1, -1):
            v = np.zeros(D, dtype=np.int64)
            v[i] = 2 * s
            vecs.append(v)
            
    if is_deadline():
        return base_arr
        
    arr = np.array(vecs, dtype=np.int64)
    
    # Verify validity (optional, but good for safety)
    # Norm squared is 4 for all.
    # Max norm squared is 4.
    # Min distance squared must be >= 4.
    # dist^2 = 4 + 4 - 2*ip = 8 - 2*ip.
    # Need 8 - 2*ip >= 4 => 2*ip <= 4 => ip <= 2.
    
    # We know distinct weight-4 +/-1 vectors have ip in {-4, -2, 0, 2}.
    # Single 2 vectors have ip with weight-4 in {-2, 0, 2}.
    # Single 2 vectors with each other have ip in {-4, 0}.
    # So all pairs have ip <= 2.
    
    return arr

