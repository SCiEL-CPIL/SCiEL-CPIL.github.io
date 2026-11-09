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
        
        # Construct the 4D vectors relative to p
        w = np.empty((m, 4), dtype=np.float64)
        w[:, 0:3] = arr - p
        w[:, 3] = arr_sq - p_sq
        
        # Use vectorized check for larger m to speed up
        if m < 25:
            for combo in itertools.combinations(range(m), 4):
                sub = w[list(combo)]
                if abs(np.linalg.det(sub)) < 0.5:
                    return False
            return True
        else:
            # Generate all 4-combinations of indices
            # For very large m, this might be slow/memory heavy, but m is expected to be < 100
            indices = np.array(list(itertools.combinations(range(m), 4)), dtype=np.int32)
            batch_size = 100000
            for i in range(0, len(indices), batch_size):
                batch_idx = indices[i:i+batch_size]
                mats = w[batch_idx]
                dets = np.linalg.det(mats)
                if np.any(np.abs(dets) < 0.5):
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
            # Shuffle skipped to avoid bias
            np.random.shuffle(skipped)
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
    
    num_restarts = 500  # Increased restarts, relying on speedup
    
    for restart in range(num_restarts):
        if time.time() > deadline - 30:  # Leave 30s margin
            break
        
        # Different strategies for ordering
        if restart % 3 == 0:
            order = grid[np.random.permutation(len(grid))]
        elif restart % 3 == 1:
            # Sort by distance from center with noise
            center = np.array([3.5, 3.5, 3.5])
            dists = np.linalg.norm(grid - center, axis=1)
            order = grid[np.argsort(dists + np.random.rand(len(grid)) * 0.1)]
        else:
            # Pure random
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
