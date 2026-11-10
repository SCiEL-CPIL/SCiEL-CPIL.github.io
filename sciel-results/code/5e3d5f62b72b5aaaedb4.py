import numpy as np
from itertools import combinations, product


def kissing_number11() -> np.ndarray:
    d = 11
    vectors = []
    
    # Generate E8 integer minimal vectors: two ±1 in positions 0..7
    e8_vectors = []
    for i, j in combinations(range(8), 2):
        for s1 in [1, -1]:
            for s2 in [1, -1]:
                v = np.zeros(8, dtype=np.int64)
                v[i] = s1
                v[j] = s2
                e8_vectors.append(v)
    
    # Generate D3 integer minimal vectors: two ±1 in positions 0..2 (will be offset to 8..10)
    d3_vectors = []
    for i, j in combinations(range(3), 2):
        for s1 in [1, -1]:
            for s2 in [1, -1]:
                v = np.zeros(3, dtype=np.int64)
                v[i] = s1
                v[j] = s2
                d3_vectors.append(v)
    
    # Take product: each point is (e8_vec * 2, d3_vec * 2)
    for e8 in e8_vectors:
        for d3 in d3_vectors:
            v = np.zeros(d, dtype=np.int64)
            v[0:8] = 2 * e8
            v[8:11] = 2 * d3
            vectors.append(v)
    
    points = np.array(vectors, dtype=np.int64)
    points = np.around(points).astype(np.int64)
    
    return points
