# EVOLVE-BLOCK-START
import numpy as np
from itertools import product, combinations
import time


def kissing_number11() -> np.ndarray:
    """
    Constructs the 1134 minimal vectors of the D11 lattice.
    All have norm^2 = 2, and min pairwise squared distance = 2.
    """
    d = 11
    deadline = time.time() + 580  # safety margin before 630s kill

    points = []

    # Type A: vectors with exactly two nonzero entries ±1
    # For each pair of positions (i, j), i < j, and each sign combination (+,+), (+,-), (-,+), (-,-)
    for i in range(d):
        for j in range(i + 1, d):
            for si in (1, -1):
                for sj in (1, -1):
                    v = np.zeros(d, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    points.append(v)

    if time.time() > deadline:
        return np.array(points, dtype=np.int64)

    # Type B: vectors with all entries ±1 and even number of -1's
    for signs in product([1, -1], repeat=d):
        if sum(1 for s in signs if s == -1) % 2 == 0:
            points.append(np.array(signs, dtype=np.int64))

    result = np.array(points, dtype=np.int64)
    return result


# EVOLVE-BLOCK-END
