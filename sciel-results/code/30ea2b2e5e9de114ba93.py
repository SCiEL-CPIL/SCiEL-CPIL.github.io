import numpy as np
import itertools
import time

D = 11

def construct():
    """
    Returns all weight-2 vectors in {-1, 0, 1}^11.
    There are C(11, 2) * 2^2 = 55 * 4 = 220 such vectors.
    They form a valid kissing configuration where min dist >= max norm.
    Max norm is sqrt(2). Min dist squared is 2 (for distinct vectors).
    sqrt(2) >= sqrt(2) holds.
    """
    candidates = []
    for combo in itertools.combinations(range(D), 2):
        for signs in itertools.product([1, -1], repeat=2):
            v = [0] * D
            for idx, s in zip(combo, signs):
                v[idx] = s
            candidates.append(v)
    
    return np.array(candidates, dtype=np.int64)
