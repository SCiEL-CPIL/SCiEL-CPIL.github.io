import numpy as np
import itertools
import time
import sys

D = 11

def construct():
    """
    Returns the set of all nonzero integer vectors in R^11 with weight 2 (entries in {-1, 1} in exactly 2 positions).
    These vectors have norm sqrt(2).
    For any two distinct such vectors u and v, the dot product u.v is in {-1, 0, 1}.
    The squared distance d(u, v)^2 = ||u||^2 + ||v||^2 - 2u.v = 2 + 2 - 2u.v = 4 - 2u.v.
    The minimum squared distance is 4 - 2(1) = 2.
    The maximum norm is sqrt(2), so squared max norm is 2.
    The condition d(u, v) >= max_norm becomes sqrt(d^2) >= sqrt(2), i.e., d^2 >= 2.
    This holds for all pairs.
    Total number of such vectors is C(11, 2) * 2^2 = 55 * 4 = 220.
    """
    vectors = []
    for i in range(D):
        for j in range(i + 1, D):
            for s1 in [1, -1]:
                for s2 in [1, -1]:
                    v = [0] * D
                    v[i] = s1
                    v[j] = s2
                    vectors.append(v)
    
    return np.array(vectors, dtype=np.int64)
