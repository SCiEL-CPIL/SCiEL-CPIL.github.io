import numpy as np
import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 135.0

    D = 11

    # Type B: Vectors with exactly two non-zero entries +/- 1
    # Norm squared = 2.
    # Number of such vectors: C(11, 2) * 2^2 = 55 * 4 = 220.
    # Let's verify the condition:
    # Min pairwise distance >= Max norm.
    # Max norm = sqrt(2).
    # Min squared distance between distinct Type B vectors:
    # 1. Disjoint supports: dist^2 = 2 + 2 - 0 = 4. (sqrt(4) = 2 > sqrt(2))
    # 2. Intersection size 1:
    #    u = (1, 1, 0...), v = (1, -1, 0...) -> u.v = -1, dist^2 = 4 - 2(-1) = 6.
    #    u = (1, 1, 0...), v = (1, 1, 0...) is not distinct.
    #    u = (1, 1, 0...), v = (-1, 1, 0...) -> u.v = -1, dist^2 = 6.
    #    u = (1, 1, 0...), v = (1, 0, 1...) -> u.v = 1, dist^2 = 4 - 2(1) = 2.
    #    sqrt(2) >= sqrt(2). OK.
    # 3. Intersection size 2 (same support):
    #    u = (1, 1, 0...), v = (1, -1, 0...) -> dist^2 = 4.
    #    u = (1, 1, 0...), v = (-1, 1, 0...) -> dist^2 = 4.
    #    u = (1, 1, 0...), v = (-1, -1, 0...) -> dist^2 = 8.
    #    All >= 2.
    
    # So the set of ALL 220 Type B vectors is a valid kissing configuration.
    
    vectors_B = []
    indices = list(range(D))
    for i, j in itertools.combinations(indices, 2):
        for s1 in [1, -1]:
            for s2 in [1, -1]:
                v = [0] * D
                v[i] = s1
                v[j] = s2
                vectors_B.append(v)
                
    final_vectors = np.array(vectors_B, dtype=np.int64)
    
    return final_vectors
