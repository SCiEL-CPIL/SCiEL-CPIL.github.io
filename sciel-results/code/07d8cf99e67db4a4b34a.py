import numpy as np

def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional points with integral coordinates.
    Uses all vectors of squared norm 2 (weight 2) in Z^11.
    There are C(11, 2) * 4 = 55 * 4 = 220 such vectors.
    For any two distinct vectors u, v in this set:
    ||u||^2 = 2, ||v||^2 = 2.
    ||u - v||^2 = 4 - 2(u . v).
    Max possible u . v for distinct weight-2 vectors is 1.
    Min ||u - v||^2 = 4 - 2(1) = 2.
    Max ||v||^2 = 2.
    Condition: Min Dist^2 >= Max Norm^2 => 2 >= 2. Holds.
    """
    d = 11
    points = []
    for i in range(d):
        for j in range(i + 1, d):
            # Signs: (1, 1), (1, -1), (-1, 1), (-1, -1)
            for s1 in [1, -1]:
                for s2 in [1, -1]:
                    p = np.zeros(d, dtype=np.int64)
                    p[i] = s1
                    p[j] = s2
                    points.append(p)
    
    points_array = np.array(points, dtype=np.int64)
    return points_array
