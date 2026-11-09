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

    Strategy: seeded random-order greedy passes over the grid, adding a point
    whenever it keeps the set sphere-free, plus a light local-improvement pass
    that tries to insert previously skipped points.  A wall-clock deadline
    guard ensures we return the best set found before the 630 s limit.

    Returns:
        points: np.ndarray of shape (k, 3) of integer grid coordinates in
            [0, n-1], no five of which are co-spherical.
    """
    n = 8
    deadline = time.monotonic() + 600.0  # hard wall-clock limit
    safety_margin = 30.0                 # stop searching this many seconds early

    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    grid_sq = (grid ** 2).sum(axis=1)  # precompute |p|^2 for the whole grid

    def can_add(arr: np.ndarray, arr_sq: np.ndarray, p: np.ndarray, p_sq: float) -> bool:
        m = len(arr)
        if m < 4:
            return True
        # Lift each existing point to a 4-vector relative to the candidate p.
        # The candidate together with four existing points is co-spherical iff
        # the corresponding four lifted vectors are linearly dependent (4x4
        # determinant == 0). Coordinates are integers, so determinants are
        # integers and a nonzero one has |det| >= 1.
        w = np.empty((m, 4), dtype=np.float64)
        w[:, 0:3] = arr - p
        w[:, 3] = arr_sq - p_sq
        combos = np.array(list(itertools.combinations(range(m), 4)))
        sub = w[combos]  # (C, 4, 4)
        return not np.any(np.abs(np.linalg.det(sub)) < 0.5)

    def local_improve(arr: np.ndarray, arr_sq: np.ndarray, skipped: list, deadline: float):
        """Try to insert previously skipped points into arr (in-place copy)."""
        arr = arr.copy()
        arr_sq = arr_sq.copy()
        improved = True
        while improved and (time.monotonic() + safety_margin < deadline):
            improved = False
            for i, p in enumerate(skipped):
                if time.monotonic() + safety_margin >= deadline:
                    break
                p_sq = grid_sq[np.where((grid == p).all(axis=1))[0][0]]
                if can_add(arr, arr_sq, p, p_sq):
                    arr = np.vstack([arr, p])
                    arr_sq = np.append(arr_sq, p_sq)
                    skipped.pop(i)
                    improved = True
        return arr, arr_sq, skipped

    rng = np.random.default_rng(0)
    best = None
    best_sq = None
    num_restarts = 50  # upper bound; actual count limited by deadline

    for restart in range(num_restarts):
        if time.monotonic() + safety_margin >= deadline:
            break
        order = grid[rng.permutation(len(grid))]
        arr = np.empty((0, 3), dtype=np.int64)
        arr_sq = np.empty(0, dtype=np.float64)
        skipped = []
        for p in order:
            if time.monotonic() + safety_margin >= deadline:
                break
            p_sq = grid_sq[np.where((grid == p).all(axis=1))[0][0]]
            if can_add(arr, arr_sq, p, p_sq):
                arr = np.vstack([arr, p]) if len(arr) else p[None, :]
                arr_sq = np.append(arr_sq, p_sq)
            else:
                skipped.append(p)

        # Light local improvement: try to insert skipped points
        arr, arr_sq, skipped = local_improve(arr, arr_sq, skipped, deadline)

        if best is None or len(arr) > len(best):
            best = arr
            best_sq = arr_sq

    if best is None:
        # Fallback: return an empty valid set (should not happen)
        best = np.empty((0, 3), dtype=np.int64)

    return best.astype(np.int64)


# EVOLVE-BLOCK-END
