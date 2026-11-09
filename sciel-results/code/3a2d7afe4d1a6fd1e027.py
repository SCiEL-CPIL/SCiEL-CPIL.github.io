# EVOLVE-BLOCK-START
import numpy as np

def kissing_number11() -> np.ndarray:
    d = 11
    pts = []

    for i in range(d):
        for j in range(i + 1, d):
            for si in (1, -1):
                for sj in (1, -1):
                    v = np.zeros(d, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    pts.append(v)

    arr = np.array(pts, dtype=np.int64)
    arr = np.around(arr).astype(np.int64)

    return arr

# EVOLVE-BLOCK-END
