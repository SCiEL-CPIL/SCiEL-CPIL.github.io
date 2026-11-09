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
    # Map from point tuple to squared norm for fast lookup
    # Since grid is small, we can just use a dictionary or rely on the array index.
    # But we already have grid_sq. We need to map point -> index in grid.
    # Or just compute p_sq on the fly? p_sq = p[0]**2 + p[1]**2 + p[2]**2. This is very fast.
    # Let's just compute it on the fly to avoid lookup overhead if we don't have a perfect hash.

    def get_p_sq(p):
        return p[0]**2 + p[1]**2 + p[2]**2

    def can_add(arr: np.ndarray, arr_sq: np.ndarray, p: np.ndarray, p_sq: float) -> bool:
        m = len(arr)
        if m < 4:
            return True
        # Lift each existing point to a 4-vector relative to the candidate p.
        # The candidate together with four existing points is co-spherical iff
        # the corresponding four lifted vectors are linearly dependent (4x4
        # determinant == 0).
        w = np.empty((m, 4), dtype=np.float64)
        w[:, 0:3] = arr - p
        w[:, 3] = arr_sq - p_sq
        
        # Vectorized batch determinant check with early exit
        # Process in chunks to manage memory
        chunk_size = 10000
        indices = list(itertools.combinations(range(m), 4))
        
        for start in range(0, len(indices), chunk_size):
            chunk_indices = indices[start:start + chunk_size]
            if not chunk_indices:
                break
            # Stack all 4x4 matrices for this chunk
            mats = w[[idx for combo in chunk_indices for idx in combo]].reshape(-1, 4, 4)
            # Compute determinants
            dets = np.linalg.det(mats)
            # Check if any determinant is too small
            if np.any(np.abs(dets) < 0.5):
                return False
        return True

    def local_improve(arr: np.ndarray, arr_sq: np.ndarray, skipped: list, deadline: float):
        """Try to insert previously skipped points into arr (in-place copy)."""
        arr = arr.copy()
        arr_sq = arr_sq.copy()
        improved = True
        # Use a local RNG for shuffling skipped
        rng_local = np.random.default_rng(42) 
        while improved and (time.monotonic() + safety_margin < deadline):
            improved = False
            # Shuffle skipped to try different orders
            rng_local.shuffle(skipped)
            i = 0
            while i < len(skipped):
                if time.monotonic() + safety_margin >= deadline:
                    break
                p, p_sq = skipped[i]
                if can_add(arr, arr_sq, p, p_sq):
                    arr = np.vstack([arr, p]) if len(arr) > 0 else p[None, :]
                    arr_sq = np.append(arr_sq, p_sq)
                    skipped.pop(i)
                    improved = True
                    # Do not increment i, check the new element at position i
                else:
                    i += 1
        return arr, arr_sq, skipped

    rng = np.random.default_rng(0)
    best = None
    best_sq = None
    num_restarts = 100  # Upper bound; actual count limited by deadline

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
            p_sq = get_p_sq(p)
            if can_add(arr, arr_sq, p, p_sq):
                arr = np.vstack([arr, p]) if len(arr) > 0 else p[None, :]
                arr_sq = np.append(arr_sq, p_sq)
            else:
                skipped.append((p, p_sq))

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
