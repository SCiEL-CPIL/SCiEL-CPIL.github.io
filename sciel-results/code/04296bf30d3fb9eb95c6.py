import numpy as np

def construct():
    D = 11
    vecs = []

    # D11 minimal vectors: exactly two nonzero entries, each +/-2
    for i in range(D):
        for j in range(i + 1, D):
            for a in (2, -2):
                for b in (2, -2):
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = a
                    v[j] = b
                    vecs.append(v)

    # E8-type vectors: eight +/-1 entries in coords 0..7, even number of -1s
    for mask in range(256):
        signs = []
        neg = 0
        for k in range(8):
            if (mask >> k) & 1:
                signs.append(-1)
                neg += 1
            else:
                signs.append(1)
        if neg % 2 == 0:
            v = np.zeros(D, dtype=np.int64)
            v[:8] = signs
            vecs.append(v)

    return np.array(vecs, dtype=np.int64)
