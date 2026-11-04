import numpy as np

D = 11

def construct():
    vectors = []
    for i in range(D):
        for j in range(D):
            if i == j:
                continue
            for si in (1, -1):
                for sj in (1, -1):
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    vectors.append(v)
    return np.array(vectors)
