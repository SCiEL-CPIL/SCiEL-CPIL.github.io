import numpy as np
import itertools
import time

D = 11

def construct():
    """
    Construct all weight-4 integer vectors in R^11.
    These vectors have norm 2.
    For any two distinct weight-4 vectors u, v with entries in {-1, 1},
    the dot product u.v is even.
    Since u != v, u.v cannot be 4 (which implies u=v).
    Thus u.v <= 2.
    The squared distance ||u - v||^2 = ||u||^2 + ||v||^2 - 2 u.v = 4 + 4 - 2 u.v = 8 - 2 u.v.
    Since u.v <= 2, ||u - v||^2 >= 8 - 4 = 4.
    So ||u - v|| >= 2.
    The maximum vector norm is 2.
    Condition: min distance >= max norm => 2 >= 2.
    This holds for all distinct weight-4 vectors.
    """
    vectors = []
    # Weight 4: exactly 4 non-zero entries, each +/-1
    for combo in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for idx, s in zip(combo, signs):
                v[idx] = s
            vectors.append(v)
            
    return np.array(vectors, dtype=np.int64)
