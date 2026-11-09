# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time


def no_5_on_a_sphere8() -> np.ndarray:
    """
    No 5 on a sphere in 8x8x8 grid.
    Maximize number of points such that no 5 are co-spherical (or co-planar).
    Uses randomized greedy + local improvement with deadline guard.
    """
    n = 8
    deadline = time.time() + 600.0
    safety_margin = 30.0  # seconds before deadline to stop

    # Precompute grid
    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    grid_sq = (grid ** 2).sum(axis=1)  # |p|^2 for each grid point

    def can_add(arr: np.ndarray, p: np.ndarray, arr_sq: np.ndarray) -> bool:
        """Check if adding p to arr keeps no 5 co-spherical."""
        m = len(arr)
        if m < 4:
            return True
        # Build the 4x4 matrix rows: (q - p, |q|^2 - |p|^2) for each q in arr
        w = np.empty((m, 4), dtype=np.float64)
        w[:, 0:3] = arr - p
        w[:, 3] = arr_sq - (p ** 2).sum()
        # Check all combinations of 4
        if m == 4:
            return abs(np.linalg.det(w)) >= 0.5
        # For m > 4, check all C(m,4) combinations
        # Use itertools for combinations
        for combo in itertools.combinations(range(m), 4):
            sub = w[list(combo)]
            if abs(np.linalg.det(sub)) < 0.5:
                return False
        return True

    def greedy_fill(order: np.ndarray, grid_sq: np.ndarray) -> tuple:
        """Greedy fill: add points in given order if feasible."""
        arr = np.empty((0, 3), dtype=np.int64)
        arr_sq = np.empty(0, dtype=np.float64)
        skipped = []
        for idx in order:
            p = grid[idx]
            if can_add(arr, p, arr_sq):
                arr = np.vstack([arr, p]) if len(arr) else p[None, :]
                arr_sq = np.append(arr_sq, grid_sq[idx])
            else:
                skipped.append(idx)
        return arr, skipped

    def local_improve(arr: np.ndarray, skipped: list, grid: np.ndarray, grid_sq: np.ndarray) -> np.ndarray:
        """Try to add skipped points back in various orders."""
        arr_sq = (arr ** 2).sum(axis=1)
        improved = False
        # Try adding skipped points one at a time in random order
        for _ in range(len(skipped)):
            if time.time() > deadline - safety_margin:
                break
            np.random.shuffle(skipped)
            new_skipped = []
            for idx in skipped:
                p = grid[idx]
                if can_add(arr, p, arr_sq):
                    arr = np.vstack([arr, p])
                    arr_sq = np.append(arr_sq, grid_sq[idx])
                    improved = True
                else:
                    new_skipped.append(idx)
            skipped = new_skipped
            if not skipped:
                break
        return arr

    np.random.seed(0)
    best = None
    best_len = 0

    # Multiple restarts with different random orderings
    num_restarts = 50
    for restart in range(num_restarts):
        if time.time() > deadline - safety_margin:
            break
        order = np.random.permutation(len(grid))
        arr, skipped = greedy_fill(order, grid_sq)

        # Local improvement
        arr = local_improve(arr, skipped, grid, grid_sq)

        if len(arr) > best_len:
            best_len = len(arr)
            best = arr.copy()

    # Final attempt: try to improve best with more aggressive local search
    if best is not None and time.time() < deadline - safety_margin:
        # Get the set of indices in best
        best_set = set(map(tuple, best))
        all_indices = set(range(len(grid)))
        not_in_best = list(all_indices - best_set)
        if not_in_best:
            best = local_improve(best, not_in_best, grid, grid_sq)

    return best.astype(np.int64)


# EVOLVE-BLOCK-END
