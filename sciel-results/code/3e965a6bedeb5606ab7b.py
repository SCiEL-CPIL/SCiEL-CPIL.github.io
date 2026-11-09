import numpy as np

def kissing_number11() -> np.ndarray:
    d = 11
    base_vectors = []
    for i in range(d):
        for j in range(i + 1, d):
            for s1 in [1, -1]:
                for s2 in [1, -1]:
                    v = np.zeros(d, dtype=np.int64)
                    v[i] = s1
                    v[j] = s2
                    base_vectors.append(v)
    
    result = np.array(base_vectors, dtype=np.int64)
    return result
