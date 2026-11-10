import numpy as np
import itertools

def kissing_number11() -> np.ndarray:
    d = 11
    # Generate all norm-2 vectors in Z^11
    # These are vectors with exactly two non-zero entries, each being +1 or -1
    vectors = []
    for i in range(d):
        for j in range(i + 1, d):
            for si in [1, -1]:
                for sj in [1, -1]:
                    v = np.zeros(d, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    vectors.append(v)
    
    result = np.array(vectors, dtype=np.int64)
    return result
