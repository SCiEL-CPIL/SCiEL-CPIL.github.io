# EVOLVE-BLOCK-START
import numpy as np


def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional points with integral coordinates corresponding to the minimal vectors of the D11 lattice.
    
    The D11 lattice consists of integer vectors with even sum of coordinates.
    The minimal vectors have exactly two non-zero coordinates, each being +/-1.
    There are binom(11, 2) * 4 = 55 * 4 = 220 such vectors.
    
    For this set:
    - Max squared norm = 2.
    - Min squared distance between distinct points = 2.
    - Condition: Min Dist^2 >= Max Norm^2 => 2 >= 2 (Valid).
    
    Returns:
        points: np.ndarray of shape (220, 11) with dtype int64.
    """
    d = 11
    points = []
    
    for i in range(d):
        for j in range(i + 1, d):
            # Generate the 4 sign combinations for positions i and j
            # 1. +1, +1
            v1 = np.zeros(d, dtype=np.int64)
            v1[i] = 1
            v1[j] = 1
            points.append(v1)
            
            # 2. +1, -1
            v2 = np.zeros(d, dtype=np.int64)
            v2[i] = 1
            v2[j] = -1
            points.append(v2)
            
            # 3. -1, +1
            v3 = np.zeros(d, dtype=np.int64)
            v3[i] = -1
            v3[j] = 1
            points.append(v3)
            
            # 4. -1, -1
            v4 = np.zeros(d, dtype=np.int64)
            v4[i] = -1
            v4[j] = -1
            points.append(v4)
            
    return np.array(points, dtype=np.int64)


# EVOLVE-BLOCK-END
