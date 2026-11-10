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

    Construction: run multiple seeded random-order greedy passes over the grid,
    adding a point whenever it keeps the set sphere-free, then apply a local
    improvement (swap) pass. Keep the largest set found. Deterministic via
    np.random.seed(0). Includes a deadline guard to stay within 600s.

    Returns:
        points: np.ndarray of shape (k, 3) of integer grid coordinates in
            [0, n-1], no five of which are co-spherical.
    """
    n = 8
    deadline = time.time() + 590  # 10s safety margin before 600s limit

    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    n_grid = len(grid)
    # Precompute squared norms for all grid points
    sq_norms = (grid ** 2).sum(axis=1).astype(np.float64)  # shape (512,)

    def can_add(arr: np.ndarray, p: np.ndarray, arr_sq: np.ndarray = None) -> bool:
        m = len(arr)
        if m < 4:
            return True
        p_sq = (p ** 2).sum()
        if arr_sq is None:
            arr_sq = (arr ** 2).sum(axis=1)
        # Build lifted vectors: for each existing point q, vector = (q - p, |q|^2 - |p|^2)
        # We need to check all C(m,4) subsets. Early exit on first violation.
        w = np.empty((m, 4), dtype=np.float64)
        w[:, 0:3] = arr - p
        w[:, 3] = arr_sq - p_sq
        # Iterate over combinations with early exit
        for combo in itertools.combinations(range(m), 4):
            sub = w[list(combo)]
            d = np.linalg.det(sub)
            if abs(d) < 0.5:
                return False
        return True

    def greedy_pass(rng: np.random.RandomState) -> tuple:
        """Run one greedy pass with random ordering. Returns (arr, arr_sq)."""
        perm = rng.permutation(n_grid)
        arr = np.empty((0, 3), dtype=np.int64)
        arr_sq = np.empty(0, dtype=np.float64)
        for idx in perm:
            if time.time() > deadline:
                break
            p = grid[idx]
            p_sq = sq_norms[idx]
            if can_add(arr, p, arr_sq):
                arr = np.vstack([arr, p]) if len(arr) else p[None, :]
                arr_sq = np.append(arr_sq, p_sq)
        return arr, arr_sq

    def local_improve(arr: np.ndarray, arr_sq: np.ndarray, rng: np.random.RandomState) -> tuple:
        """Try to improve the set by removing one point and greedily extending."""
        if len(arr) == 0:
            return arr, arr_sq
        improved = True
        while improved and time.time() < deadline:
            improved = False
            m = len(arr)
            # Try removing each point and greedily extend the rest
            for i in range(m):
                if time.time() > deadline:
                    break
                # Build set without point i
                arr_without_i = np.delete(arr, i, axis=0)
                arr_sq_without_i = np.delete(arr_sq, i)
                # Now greedily try to add points not in arr_without_i
                new_arr = arr_without_i.copy()
                new_arr_sq = arr_sq_without_i.copy()
                # Use a set for O(1) membership
                current_set = {tuple(row) for row in new_arr}
                # Try grid points in random order
                order = rng.permutation(n_grid)
                for idx in order:
                    if time.time() > deadline:
                        break
                    p = grid[idx]
                    pt = tuple(p)
                    if pt in current_set:
                        continue
                    p_sq = sq_norms[idx]
                    if can_add(new_arr, p, new_arr_sq):
                        new_arr = np.vstack([new_arr, p])
                        new_arr_sq = np.append(new_arr_sq, p_sq)
                        current_set.add(pt)
                if len(new_arr) > m:
                    arr = new_arr
                    arr_sq = new_arr_sq
                    improved = True
                    break
        return arr, arr_sq

    np.random.seed(0)
    rng = np.random.RandomState(0)
    best = None
    best_sq = None
    best_len = 0

    # Run multiple greedy passes with local improvement
    num_passes = 0
    while time.time() < deadline - 30:
        num_passes += 1
        arr, arr_sq = greedy_pass(rng)
        if len(arr) > 0:
            arr, arr_sq = local_improve(arr, arr_sq, rng)
        if best is None or len(arr) > best_len:
            best = arr
            best_sq = arr_sq
            best_len = len(arr)
        # Early exit if we've found a good solution and time is running low
        if best_len >= 50 and num_passes >= 15 and time.time() > deadline - 60:
            break
        # Also break if we've done many passes with no improvement
        if num_passes >= 40 and best_len < 45:
            break

    if best is None:
        # Fallback: just return first few points that work
        best = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.int64)

    return best.astype(np.int64)


# EVOLVE-BLOCK-END
