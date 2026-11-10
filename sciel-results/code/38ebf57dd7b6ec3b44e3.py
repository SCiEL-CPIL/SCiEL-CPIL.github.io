# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time
import functools


def no_5_on_a_sphere8() -> np.ndarray:
    """
    Maximize the number of points from the 8x8x8 integer grid such that no
    five selected points are co-spherical (which also forbids five co-planar).
    Uses vectorized greedy construction plus swap-based local improvement,
    with an explicit deadline guard.
    """
    start_time = time.time()
    DEADLINE = 630.0  # wall-clock limit; return before this with margin

    n = 8
    grid = np.array(list(itertools.product(range(n), range(n), range(n))),
                    dtype=np.int64)  # (512, 3)
    N = len(grid)

    def remaining() -> float:
        return DEADLINE - 0.2 - time.time()

    @functools.lru_cache(maxsize=None)
    def get_combos(m):
        """Cache combinations of indices for 4 points from m points."""
        if m < 4:
            return np.empty((0, 4), dtype=np.int64)
        return np.array(list(itertools.combinations(range(m), 4)), dtype=np.int64)

    def is_valid_set(arr: np.ndarray) -> bool:
        """Check if the entire set is valid (no 5 co-spherical/co-planar)."""
        m = len(arr)
        if m < 5:
            return True
        arr_f = arr.astype(np.float64)
        arr_sq = (arr_f ** 2).sum(axis=1)
        
        # We need to check all 5-subsets.
        # It's more efficient to check if adding each point to the previous 4 is valid?
        # No, that only checks subsets where the 'new' point is the last one in the subset.
        # But any 5 points can be ordered such that one is 'last'.
        # So checking all (p, q1..q4) where p is any point and q1..q4 are any 4 others?
        # That's $\binom{m}{5} \times 5$ checks? No.
        # If we check for each p in arr, and each 4-combo of (arr \ {p}), that covers all 5-subsets 5 times.
        # It's redundant but correct.
        
        # More efficient: 
        # For each i, check if arr[i] is co-spherical with any 4 from arr[:i] U arr[i+1:]?
        # Actually, the incremental `can_add` logic works if we build the set.
        # For a static check, we can just iterate all 5-combos.
        
        # Given m is likely < 40, $\binom{40}{5} \approx 658008$.
        # 650k determinants is fast in NumPy if vectorized.
        
        combos_5 = np.array(list(itertools.combinations(range(m), 5)), dtype=np.int64)
        if len(combos_5) == 0:
            return True
            
        # For each 5-combo, we need to check if they are co-spherical.
        # A set of 5 points is co-spherical iff the determinant of the 5x5 matrix
        # [x y z x^2+y^2+z^2 1] is 0?
        # Or using the 4x4 check: 
        # Pick one point as reference p. Check det of 4x4 for other 4.
        # If ANY 4x4 det is 0, they are co-spherical.
        # Actually, if 5 points are co-spherical, ALL 5 choices of reference point will yield 0 det.
        # If they are NOT co-spherical, NO choice yields 0 det?
        # Wait. If 5 points are co-spherical, they lie on a sphere.
        # The equation of a sphere is $x^2+y^2+z^2 + Ax + By + Cz + D = 0$.
        # For 5 points, we have 5 equations for A,B,C,D.
        # If the system has a solution, they are co-spherical.
        # The condition for solvability is that the augmented matrix has rank < 5?
        # The matrix is:
        # [ x_i y_i z_i x_i^2+y_i^2+z_i^2 1 ]
        # If det of this 5x5 matrix is 0, they are co-spherical.
        
        # Let's use the 5x5 determinant check. It's cleaner.
        # Rows: [x, y, z, x^2+y^2+z^2, 1]
        
        pts = arr_f[combos_5] # (C, 5, 3)
        sq = (pts ** 2).sum(axis=2) # (C, 5)
        
        M = np.empty((len(combos_5), 5, 5), dtype=np.float64)
        M[:, :, 0:3] = pts
        M[:, :, 3] = sq
        M[:, :, 4] = 1.0
        
        dets = np.linalg.det(M)
        return not np.any(np.abs(dets) < 0.5)

    def can_add(arr: np.ndarray, p: np.ndarray) -> bool:
        """True if adding p to arr keeps the set sphere-free."""
        m = len(arr)
        if m < 4:
            return True
        # Lift matrix: rows are (q - p, |q|^2 - |p|^2) for q in arr.
        w = np.empty((m, 4), dtype=np.float64)
        arr_f = arr.astype(np.float64)
        p_f = p.astype(np.float64)
        w[:, 0:3] = arr_f - p_f
        w[:, 3] = (arr_f ** 2).sum(axis=1) - (p_f ** 2).sum()
        
        combos = get_combos(m)
        if len(combos) == 0:
            return True
        sub = w[combos]  # (C, 4, 4)
        return not np.any(np.abs(np.linalg.det(sub)) < 0.5)

    def can_add_batch(arr: np.ndarray, candidates: np.ndarray) -> np.ndarray:
        """Vectorized: for each candidate, can it be added to arr? Returns bool array."""
        m = len(arr)
        if m < 4:
            return np.ones(len(candidates), dtype=bool)
        # Precompute for existing points: coordinates and squared norms.
        arr_f = arr.astype(np.float64)
        arr_sq = (arr_f ** 2).sum(axis=1)  # (m,)
        cand_f = candidates.astype(np.float64)
        cand_sq = (cand_f ** 2).sum(axis=1)  # (B,)
        # Build w for each candidate: (B, m, 4)
        # w[b, j, 0:3] = arr[j] - cand[b]; w[b, j, 3] = arr_sq[j] - cand_sq[b]
        B = len(candidates)
        w = np.empty((B, m, 4), dtype=np.float64)
        w[:, :, 0:3] = arr_f[None, :, :] - cand_f[:, None, :]
        w[:, :, 3] = arr_sq[None, :] - cand_sq[:, None, None]
        
        combos = get_combos(m)
        if len(combos) == 0:
            return np.ones(B, dtype=bool)
            
        sub = w[:, combos, :, :]  # (B, C, 4, 4)
        dets = np.linalg.det(sub)  # (B, C)
        return not np.any(np.abs(dets) < 0.5, axis=1)

    def greedy_pass(order: np.ndarray) -> np.ndarray:
        """Greedy: add points in given order if they stay valid."""
        arr = np.empty((0, 3), dtype=np.int64)
        for idx in order:
            p = grid[idx]
            if can_add(arr, p):
                if len(arr) == 0:
                    arr = p[None, :]
                else:
                    arr = np.vstack([arr, p])
        return arr

    def local_improve(arr: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
        """Try swaps (remove one, add one) to grow the set. Iterate until no gain."""
        arr = arr.copy()
        improved = True
        while improved and remaining() > 5.0:
            improved = False
            m = len(arr)
            # For each point to remove, try to find a candidate to add that grows size.
            for i in range(m):
                if remaining() < 5.0:
                    break
                without_i = np.delete(arr, i, axis=0)
                # Candidates: all grid points not already in without_i.
                in_set = np.isin(grid, without_i, axis=1)
                candidates = grid[~in_set]
                if len(candidates) == 0:
                    continue
                # Check in batches to be safe.
                ok = can_add_batch(without_i, candidates)
                if np.any(ok):
                    # Pick the first valid candidate (or a random one for diversity).
                    valid_idx = np.where(ok)[0]
                    pick = valid_idx[rng.randint(len(valid_idx))]
                    new_arr = np.vstack([without_i, candidates[pick]])
                    # This keeps size the same (swap). For growth we need a point
                    # that can be added WITHOUT removing. Handle below.
                    improved = True
                    arr = new_arr
                    break
            # Also try pure additions (grow) if possible.
            if remaining() < 5.0:
                break
            in_set = np.isin(grid, arr, axis=1)
            candidates = grid[~in_set]
            if len(candidates) > 0:
                ok = can_add_batch(arr, candidates)
                if np.any(ok):
                    valid_idx = np.where(ok)[0]
                    pick = valid_idx[rng.randint(len(valid_idx))]
                    arr = np.vstack([arr, candidates[pick]])
                    improved = True

        return arr

    # Multi-restart greedy + local improvement.
    rng = np.random.RandomState(0)
    best = None
    best_len = -1

    # Generate a valid fallback first
    fallback = None
    for _ in range(100):
        # Try to build a small valid set
        idxs = rng.choice(N, size=10, replace=False)
        cand = grid[idxs]
        if is_valid_set(cand):
            fallback = cand
            break
    if fallback is None:
        # Extremely unlikely, but generate a very sparse set
        # Points with large spacing are less likely to be co-spherical
        # Just pick 5 points that are definitely not co-spherical
        # (0,0,0), (1,0,0), (0,1,0), (0,0,1), (2,2,2)
        # Check manually?
        # For now, return a set of 1 point if all else fails (trivially valid)
        fallback = grid[:1]

    best = fallback
    best_len = len(fallback)

    num_restarts = 30
    for r in range(num_restarts):
        if remaining() < 20.0:
            break
        order = rng.permutation(N)
        arr = greedy_pass(order)
        # Verify the greedy result before improving?
        # Greedy ensures validity by construction.
        arr = local_improve(arr, rng)
        if len(arr) > best_len:
            # Verify before accepting to be safe
            if is_valid_set(arr):
                best_len = len(arr)
                best = arr
            else:
                # Should not happen if can_add is correct, but handle it
                pass

    # Final polish: one more local-improvement pass on the best found.
    if best is not None and remaining() > 10.0:
        polished = local_improve(best, rng)
        if len(polished) > best_len:
            if is_valid_set(polished):
                best = polished
                best_len = len(polished)

    return best.astype(np.int64)


# EVOLVE-BLOCK-END
