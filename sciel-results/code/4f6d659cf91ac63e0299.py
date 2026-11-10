# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time


def no_5_on_a_sphere8() -> np.ndarray:
    """
    The "no 5 on a sphere" problem. Select as many points as possible from the
    n x n x n integer grid {0,...,n-1}^3 such that no five of the selected points
    are co-spherical (lie on a common sphere, which also forbids five co-planar
    points).

    Here n = 8. The objective is to MAXIMIZE the number of selected points.

    Uses randomized greedy search with multiple restarts and time management.
    """
    n = 8
    deadline = time.time() + 620  # Safety margin before 630s limit

    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)

    def can_add(arr: np.ndarray, p: np.ndarray) -> bool:
        m = len(arr)
        if m < 4:
            return True
        w = np.empty((m, 4), dtype=np.float64)
        w[:, 0:3] = arr - p
        w[:, 3] = (arr ** 2).sum(axis=1) - (p ** 2).sum()
        for combo in itertools.combinations(range(m), 4):
            if abs(np.linalg.det(w[list(combo)])) < 0.5:
                return False
        return True

    np.random.seed(0)
    best = None
    best_len = 0

    for restart in range(200):  # Upper bound on restarts
        if time.time() > deadline - 15:  # Stop if less than 15s remaining
            break
        
        order = grid[np.random.permutation(len(grid))]
        arr = np.empty((0, 3), dtype=np.int64)
        
        for p in order:
            if time.time() > deadline - 15:
                break
            if can_add(arr, p):
                arr = np.vstack([arr, p]) if len(arr) else p[None, :]
        
        if best is None or len(arr) > best_len:
            best = arr.copy()
            best_len = len(arr)

    return best.astype(np.int64) if best is not None else np.empty((0, 3), dtype=np.int64)


# EVOLVE-BLOCK-END
