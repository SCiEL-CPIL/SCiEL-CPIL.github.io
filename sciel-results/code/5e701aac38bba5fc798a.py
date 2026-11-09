# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations
import time


def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional integer points forming a valid kissing configuration.
    Uses the D11 lattice minimal vectors (weight 2, entries ±1) for 220 points with norm squared 2,
    and mutual squared distances at least 2.
    
    Returns:
        points: np.ndarray of shape (220, 11)
    """
    d = 11
    vectors = []
    
    # Generate all D11 minimal vectors: exactly two non-zero entries, each ±1
    for i, j in combinations(range(d), 2):
        for si in [1, -1]:
            for sj in [1, -1]:
                v = np.zeros(d, dtype=np.int64)
                v[i] = si
                v[j] = sj
                vectors.append(v)
    
    points = np.array(vectors, dtype=np.int64)
    
    # Convert to rounded int64 array as per task requirements
    points = np.around(points).astype(np.int64)
    
    return points


# EVOLVE-BLOCK-END
