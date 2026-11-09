# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations, product


def kissing_number11() -> np.ndarray:
    """
    Constructs a kissing configuration in dimension 11 using a direct
    integer lattice construction combining scaled E8 and D3 minimal vectors.

    E8 has 240 minimal vectors with squared norm 2. Scaled by 2, they have
    squared norm 8 and min squared distance 8.
    D3 has 12 minimal vectors with squared norm 2. Scaled by 2, they have
    squared norm 8 and min squared distance 8.

    The union satisfies min_dist^2 >= max_norm^2 (8 >= 8).
    Total points: 240 + 12 = 252.

    Returns:
        points: np.ndarray of shape (252, 11) with int64 entries
    """
    d = 11
    vectors = []

    # Part 1: 2 * E8 in dimensions 0..7
    # Type 1: +-2, +-2 at two positions
    for i, j in combinations(range(8), 2):
        for s1 in [1, -1]:
            for s2 in [1, -1]:
                v = np.zeros(d, dtype=np.int64)
                v[i] = 2 * s1
                v[j] = 2 * s2
                vectors.append(v)

    # Type 2: +-1 at all 8 positions, even number of -1s
    for signs in product([1, -1], repeat=8):
        if sum(1 for s in signs if s == -1) % 2 == 0:
            v = np.zeros(d, dtype=np.int64)
            for k in range(8):
                v[k] = signs[k]
            vectors.append(v)

    # Part 2: 2 * D3 in dimensions 8..10
    # +-2, +-2 at two positions in the 3D subspace
    for i, j in combinations([8, 9, 10], 2):
        for s1 in [1, -1]:
            for s2 in [1, -1]:
                v = np.zeros(d, dtype=np.int64)
                v[i] = 2 * s1
                v[j] = 2 * s2
                vectors.append(v)

    points = np.array(vectors, dtype=np.int64)
    points = np.around(points).astype(np.int64)

    return points


# EVOLVE-BLOCK-END
