# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time


def no_5_on_a_sphere8() -> np.ndarray:
    """
    The "no 5 on a sphere" problem. Select as many points as possible from the
    8x8x8 integer grid {0,...,7}^3 such that no five of the selected points
    are co-spherical (lie on a common sphere, which also forbids five co-planar
    points).

    Strategy:
    1. Seeded random-order greedy passes over the grid, adding a point whenever
       it keeps the set sphere-free.
    2. A local-improvement pass that tries to insert previously skipped points.
    3. A wall-clock deadline guard ensures we return the best set found before
       the 630 s limit.

    Returns:
        points: np.ndarray of shape (k, 3) of integer grid coordinates in
            [0, 7], no five of which are co-spherical.
    """
    n = 8
    deadline = time.time() + 600  # 600 second limit, with margin

    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    grid_sq = (grid ** 2).sum(axis=1)  # precompute squared norms

    def can_add(arr: np.ndarray, p: np.ndarray, arr_sq: np.ndarray = None) -> bool:
        """Check if point p can be added to arr without creating 5 co-spherical points."""
        m = len(arr)
        if m < 4:
            return True
        
        if arr_sq is None:
            arr_sq = (arr ** 2).sum(axis=1)
        
        p_sq = (p ** 2).sum()
        
        # For each combination of 4 existing points, check if p makes them co-spherical
        # We need to check: for every 4 existing points q1..q4, the 4x4 matrix
        # with rows (qi - p, |qi|^2 - |p|^2) has nonzero determinant
        # Equivalently: the 5 points p, q1, q2, q3, q4 are co-spherical iff
        # det of 5x5 matrix with rows [xi, yi, zi, xi^2+yi^2+zi^2, 1] is 0
        
        # More efficient: lift to 4D. Points are co-spherical iff their lifted
        # vectors are linearly dependent.
        # Lifted vector: (x, y, z, x^2+y^2+z^2 - 1) or similar
        # Actually, standard: (x, y, z, x^2+y^2+z^2, 1) -> 5x5 det
        # But we can use: relative to p, the condition is that (qi - p, |qi|^2 - |p|^2)
        # for i=1..4 are linearly dependent in R^4
        
        w = np.empty((m, 4), dtype=np.float64)
        w[:, 0:3] = arr - p
        w[:, 3] = arr_sq - p_sq
        
        # Check all combinations of 4 rows
        for combo in itertools.combinations(range(m), 4):
            sub = w[list(combo)]
            if abs(np.linalg.det(sub)) < 0.5:
                return False
        return True

    def greedy_fill(order: np.ndarray) -> np.ndarray:
        """Greedy fill with given order."""
        arr = np.empty((0, 3), dtype=np.int64)
        arr_sq = np.empty(0, dtype=np.float64)
        for p in order:
            if can_add(arr, p, arr_sq):
                if len(arr) == 0:
                    arr = p[None, :]
                    arr_sq = (p ** 2).sum()
                else:
                    arr = np.vstack([arr, p])
                    arr_sq = np.append(arr_sq, (p ** 2).sum())
        return arr

    def local_improve(arr: np.ndarray, skipped: np.ndarray) -> np.ndarray:
        """Try to insert skipped points into arr."""
        if len(skipped) == 0:
            return arr
        
        arr_sq = (arr ** 2).sum(axis=1)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            new_skipped = []
            for p in skipped:
                if can_add(arr, p, arr_sq):
                    if len(arr) == 0:
                        arr = p[None, :]
                        arr_sq = (p ** 2).sum()
                    else:
                        arr = np.vstack([arr, p])
                        arr_sq = np.append(arr_sq, (p ** 2).sum())
                    improved = True
                else:
                    new_skipped.append(p)
            skipped = np.array(new_skipped, dtype=np.int64) if new_skipped else np.empty((0, 3), dtype=np.int64)
        return arr

    np.random.seed(0)
    best = None
    best_len = 0
    
    num_restarts = 200  # More restarts, but with deadline guard
    
    for restart in range(num_restarts):
        if time.time() > deadline - 30:  # Leave 30s margin
            break
        
        order = grid[np.random.permutation(len(grid))]
        
        # Greedy fill
        arr = greedy_fill(order)
        
        # Local improvement: try to add skipped points
        in_arr = set(map(tuple, arr))
        skipped = [p for p in order if tuple(p) not in in_arr]
        if skipped:
            skipped_arr = np.array(skipped, dtype=np.int64)
            arr = local_improve(arr, skipped_arr)
        
        if best is None or len(arr) > best_len:
            best = arr.copy()
            best_len = len(arr)
    
    # Final verification and return
    if best is None:
        best = grid[:10]  # Fallback
    
    return best.astype(np.int64)


# EVOLVE-BLOCK-END
