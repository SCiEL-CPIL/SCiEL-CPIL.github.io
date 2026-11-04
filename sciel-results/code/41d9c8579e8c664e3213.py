import numpy as np

def construct():
    D = 11
    vectors = []
    for i in range(D):
        for j in range(i + 1, D):
            # 4 combinations of signs for positions i and j
            for si in (1, -1):
                for sj in (1, -1):
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    vectors.append(v)
    return np.array(vectors)
