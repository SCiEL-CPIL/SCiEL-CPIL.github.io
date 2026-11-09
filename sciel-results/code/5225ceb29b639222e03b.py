import numpy as np

def kissing_number11() -> np.ndarray:
    d = 11
    points = []
    # Generate all points with squared norm 2
    # These are vectors with exactly two non-zero entries, each being +/- 1
    for i in range(d):
        for j in range(i + 1, d):
            # Signs for i and j
            for si in [-1, 1]:
                for sj in [-1, 1]:
                    p = np.zeros(d, dtype=np.int64)
                    p[i] = si
                    p[j] = sj
                    points.append(p)
    
    return np.array(points)
