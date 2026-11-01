import numpy as np
import itertools

def construct():
    D = 11
    candidates = []
    
    # Type 1: two ±1's, rest 0 (norm^2 = 2)
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i] = si
                v[j] = sj
                candidates.append(v)
                
    candidates = np.array(candidates, dtype=np.int64)
    
    # Verify validity just in case
    # Max norm is sqrt(2). Min dist must be >= sqrt(2).
    # Dist sq >= 2.
    # For distinct Type 1 vectors, min dist sq is 2.
    # So it is valid.
    
    return candidates
