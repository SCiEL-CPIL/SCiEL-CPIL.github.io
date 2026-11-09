# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations


def kissing_number11() -> np.ndarray:
    """
    Constructs 220 eleven-dimensional integer points (all vectors with exactly
    two non-zero entries, each +/-1) satisfying:
      - shape (k, 11)
      - k >= 2
      - no zero vector
      - min pairwise squared distance >= max squared norm
    """
    d = 11
    points_list = []

    # Generate all vectors with exactly 2 non-zero entries, each being +/-1
    for i, j in combinations(range(d), 2):
        for si in (1, -1):
            for sj in (1, -1):
                p = np.zeros(d, dtype=np.int64)
                p[i] = si
                p[j] = sj
                points_list.append(p)

    points = np.array(points_list, dtype=np.int64)
    points = np.around(points).astype(np.int64)

    return points


# EVOLVE-BLOCK-END
