"""
Constructs a kissing configuration in R^11 using all integer vectors of squared norm 8.
The vectors are of two types:
1. Two coordinates are +/-2, rest 0.
2. Eight coordinates are +/-1, rest 0. To maintain min distance >= sqrt(8), we must restrict
   the sign patterns for each support to either all even-parity or all odd-parity sets.
   We choose even-parity (even number of -1s) for all supports.
"""
import numpy as np
from itertools import combinations

def construct():
    D = 11
    vectors = []
    
    # Type 1: Two coordinates +/-2
    # Number of vectors: C(11, 2) * 4
    for i, j in combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i] = 2 * si
                v[j] = 2 * sj
                vectors.append(v)
                
    # Type 2: Eight coordinates +/-1
    # Number of vectors: C(11, 8) * 2^(8-1)
    # We iterate over all subsets of 8 indices.
    # For each subset, we generate all sign patterns with an even number of -1s.
    for support in combinations(range(D), 8):
        # Generate all 2^8 sign combinations
        # Mask from 0 to 255
        for mask in range(256):
            minus_count = 0
            v = np.zeros(D, dtype=np.int64)
            for k, idx in enumerate(support):
                if (mask >> k) & 1:
                    v[idx] = -1
                    minus_count += 1
                else:
                    v[idx] = 1
            # Check even parity (even number of -1s)
            if minus_count % 2 == 0:
                vectors.append(v)
                
    return np.array(vectors, dtype=np.int64)
