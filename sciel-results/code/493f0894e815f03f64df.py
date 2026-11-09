import numpy as np

def kissing_number11() -> np.ndarray:
    """
    Constructs 220 points in Z^11 with norm^2 = 2.
    Each point has exactly two non-zero entries (each ±1).
    Min pairwise dist^2 = 2, max norm^2 = 2. Valid since 2 >= 2.
    """
    d = 11
    points = []
    for i in range(d):
        for j in range(i + 1, d):
            for si in [1, -1]:
                for sj in [1, -1]:
                    v = np.zeros(d, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    points.append(v)
    
    return np.array(points)
