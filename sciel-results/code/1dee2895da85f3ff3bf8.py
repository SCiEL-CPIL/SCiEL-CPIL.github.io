"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Constructs a kissing configuration using integer vectors of norm sqrt(2).
    Vectors are of the form +/-e_i +/-e_j.
    Condition: min pairwise distance >= max norm.
    For norm sqrt(2), max norm M = sqrt(2).
    Distance squared between distinct vectors v, w:
    If they share 1 index, dot product is +/-1. Dist^2 = 2+2-2(+/-1) = 4 or 0?
    Wait.
    v = e1+e2, w = e1+e3. dot = 1. Dist^2 = 2+2-2(1) = 2. Dist = sqrt(2).
    sqrt(2) >= sqrt(2). OK.
    v = e1+e2, w = e1-e3. dot = 1. Dist^2 = 2. OK.
    v = e1+e2, w = -e1+e3. dot = -1. Dist^2 = 2+2-2(-1) = 6. OK.
    v = e1+e2, w = e3+e4. dot = 0. Dist^2 = 4. OK.
    v = e1+e2, w = -e1-e2. dot = -2. Dist^2 = 2+2-2(-2) = 8. OK.
    
    So all vectors of the form +/-e_i +/-e_j are mutually compatible.
    Total count: C(11, 2) * 4 = 55 * 4 = 220.
    """
    vectors = []
    for i in range(D):
        for j in range(i + 1, D):
            for si in (1, -1):
                for sj in (1, -1):
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    vectors.append(v)
    
    if len(vectors) > 0:
        return np.array(vectors, dtype=np.int64)
    else:
        # Fallback: return empty or single vector
        return np.array([[1] + [0]*(D-1)], dtype=np.int64)
# EVOLVE-BLOCK-END
