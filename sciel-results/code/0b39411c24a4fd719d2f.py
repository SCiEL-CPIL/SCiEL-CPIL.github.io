import numpy as np

def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional points with integral coordinates.
    Uses all 220 vectors with exactly two non-zero entries of value +/-1.
    Max squared norm is 2.
    Min squared distance is 2.
    Condition min_dist_sq >= max_norm_sq holds (2 >= 2).
    """
    d = 11
    points = []
    
    # Generate all pairs of indices
    for i in range(d):
        for j in range(i + 1, d):
            # 4 sign combinations for the two non-zero entries
            # v[i] = s1, v[j] = s2, others 0
            for s1 in [1, -1]:
                for s2 in [1, -1]:
                    v = np.zeros(d, dtype=np.int64)
                    v[i] = s1
                    v[j] = s2
                    points.append(v)
    
    points_array = np.array(points, dtype=np.int64)
    # Ensure shape is (k, 11)
    # k should be 220
    return points_array
