"""Structural change: Constructive enumeration of all weight-2 integer vectors in R^11.
The set of all 220 weight-2 vectors (norm^2=2) forms a valid kissing configuration
where the minimum pairwise distance is sqrt(2) and the maximum norm is sqrt(2).
This is a provably valid solution of size 220, which is superior to the greedy
search results (approx 213) and runs in constant time.
"""
import numpy as np
import itertools


def construct():
    D = 11
    vectors = []
    
    # Generate all weight-2 vectors (exactly two non-zero entries, each +/-1)
    # There are C(11, 2) * 2^2 = 55 * 4 = 220 such vectors.
    for positions in itertools.combinations(range(D), 2):
        for signs in itertools.product([1, -1], repeat=2):
            v = [0] * D
            v[positions[0]] = signs[0]
            v[positions[1]] = signs[1]
            vectors.append(tuple(v))
            
    # Convert to numpy array
    result = np.array(vectors, dtype=np.int64)
    return result
