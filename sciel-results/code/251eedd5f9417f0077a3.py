import numpy as np
import time

def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional points with integral coordinates.
    Uses the set of all vectors with exactly two non-zero entries (+/- 1).
    This yields 220 points, which is a valid kissing configuration.
    """
    d = 11
    points = []
    
    # Generate all vectors with exactly two non-zero entries
    for i in range(d):
        for j in range(i + 1, d):
            for si in [-1, 1]:
                for sj in [-1, 1]:
                    v = np.zeros(d, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    points.append(v)
                    
    # Convert to array
    points_array = np.array(points, dtype=np.int64)
    
    return points_array
