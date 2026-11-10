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
    deadline = time.time() + 580  # 20s safety margin before 600s limit

    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    n_grid = len(grid)

    def can_add(arr: np.ndarray, p: np.ndarray) -> bool:
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
        w[:, 3] = (arr ** 2).sum(axis=1) - (p ** 2).sum()
        combos = np.array(list(itertools.combinations(range(m), 4)))
        sub = w[combos]  # (C, 4, 4)
        return not np.any(np.abs(np.linalg.det(sub)) < 0.5)

    def greedy_pass(rng: np.random.RandomState) -> np.ndarray:
        """Run one greedy pass with random ordering."""
        order = grid[rng.permutation(n_grid)]
        arr = np.empty((0, 3), dtype=np.int64)
        for p in order:
            if time.time() > deadline:
                break
            if can_add(arr, p):
                arr = np.vstack([arr, p]) if len(arr) else p[None, :]
        return arr

    def local_improve(arr: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
        """Try to improve the set by swapping points that allow adding others."""
        if len(arr) == 0:
            return arr
        improved = True
        while improved and time.time() < deadline:
            improved = False
            m = len(arr)
            # Try removing each point and see if we can add a better one
            for i in range(m):
                if time.time() > deadline:
                    break
                # Try to find a point not in arr that can be added after removing arr[i]
                arr_without_i = np.delete(arr, i, axis=0)
                # Try a few random candidates
                for _ in range(5):
                    p = grid[rng.randint(n_grid)]
                    if p in arr_without_i:
                        continue
                    if can_add(arr_without_i, p):
                        # Check if this actually helps (we can add more later)
                        new_arr = np.vstack([arr_without_i, p])
                        # Try to extend new_arr greedily with remaining points
                        remaining = [q for q in grid if q not in new_arr]
                        for q in remaining[:20]:  # limit to keep fast
                            if can_add(new_arr, q):
                                new_arr = np.vstack([new_arr, q])
                                break
                        if len(new_arr) > m:
                            arr = new_arr
                            improved = True
                            break
                if improved:
                    break
        return arr

    np.random.seed(0)
    rng = np.random.RandomState(0)
    best = None

    # Run multiple greedy passes
    num_restarts = 20
    for restart in range(num_restarts):
        if time.time() > deadline - 30:  # leave time for polish
            break
        arr = greedy_pass(rng)
        if len(arr) > 0:
            # Apply local improvement
            arr = local_improve(arr, rng)
        if best is None or len(arr) > len(best):
            best = arr

    # Final: try a few more random restarts if time allows
    while time.time() < deadline - 5:
        arr = greedy_pass(rng)
        if len(arr) > 0:
            arr = local_improve(arr, rng)
        if best is None or len(arr) > len(best):
            best = arr

    if best is None:
        # Fallback: just return first few points that work
        best = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.int64)

    return best.astype(np.int64)


# EVOLVE-BLOCK-END
