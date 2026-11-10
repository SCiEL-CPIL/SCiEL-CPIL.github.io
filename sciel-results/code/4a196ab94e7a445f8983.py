# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time


def no_5_on_a_sphere8() -> np.ndarray:
    """
    The "no 5 on a sphere" problem. Select as many points as possible from the
    n x n x n integer grid {0,...,n-1}^3 such that no five of the selected points
    are co-spherical (lie on a common sphere, which also forbids five co-planar
    points). Five points p_i = (x_i, y_i, z_i) are co-spherical iff the 5x5
    determinant of rows [x_i, y_i, z_i, x_i^2 + y_i^2 + z_i^2, 1] vanishes.

    Here n = 8. The objective is to MAXIMIZE the number of selected points.

    Uses randomized greedy search with multiple restarts and a time deadline
    guard to stay within evaluation limits.

    Returns:
        points: np.ndarray of shape (k, 3) of integer grid coordinates in
            [0, n-1], no five of which are co-spherical.
    """
    n = 8
    deadline = time.time() + 570  # 570s to leave 30s safety margin

    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    grid_sq = (grid ** 2).sum(axis=1)  # Precompute |p|^2 for all grid points

    def can_add(arr: np.ndarray, arr_sq: np.ndarray, p: np.ndarray, p_sq: float) -> bool:
        m = len(arr)
        if m < 4:
            return True
        # Lift each existing point to a 4-vector relative to the candidate p.
        w = np.empty((m, 4), dtype=np.float64)
        w[:, 0:3] = arr - p
        w[:, 3] = arr_sq - p_sq
        combos = np.array(list(itertools.combinations(range(m), 4)))
        sub = w[combos]  # (C, 4, 4)
        dets = np.linalg.det(sub)
        return not np.any(np.abs(dets) < 0.5)

    np.random.seed(0)
    best = None
    best_len = 0
    restart = 0
    while time.time() < deadline:
        restart += 1
        order = grid[np.random.permutation(len(grid))]
        arr = np.empty((0, 3), dtype=np.int64)
        arr_sq = np.empty(0, dtype=np.int64)
        for p in order:
            if time.time() > deadline:
                break
            p_sq = (p ** 2).sum()
            if can_add(arr, arr_sq, p, p_sq):
                if len(arr) == 0:
                    arr = p[None, :]
                    arr_sq = np.array([p_sq])
                else:
                    arr = np.vstack([arr, p])
                    arr_sq = np.append(arr_sq, p_sq)
        if best is None or len(arr) > best_len:
            best = arr
            best_len = len(arr)
        # If we've found a very good solution, we can stop early
        if best_len >= 50:
            break

    if best is None:
        # Fallback: return empty or minimal valid set
        return np.empty((0, 3), dtype=np.int64)

    return best.astype(np.int64)


# EVOLVE-BLOCK-END
