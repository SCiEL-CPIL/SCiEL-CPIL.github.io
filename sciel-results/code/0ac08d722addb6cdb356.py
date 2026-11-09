import numpy as np
import time

START_TIME = time.time()
TIME_LIMIT = 590

def kissing_number11() -> np.ndarray:
    """
    Returns a 2-D array of shape (k, 11) with integer entries (int64).
    Uses the set of all integer vectors in Z^11 with exactly two non-zero entries (each +/-1).
    k = C(11, 2) * 2^2 = 55 * 4 = 220.
    """
    if time.time() - START_TIME > TIME_LIMIT:
        # Fallback
        d = 11
        points = np.array([np.ones(d), -1 * np.ones(d)]).astype(np.int64)
        return points

    d = 11
    points = []
    for i in range(d):
        for j in range(i + 1, d):
            for si in (-1, 1):
                for sj in (-1, 1):
                    v = np.zeros(d, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    points.append(v)
    
    arr = np.array(points, dtype=np.int64)
    arr = np.around(arr).astype(np.int64)
    return arr
