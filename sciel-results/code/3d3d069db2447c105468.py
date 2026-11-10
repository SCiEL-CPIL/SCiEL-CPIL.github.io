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

    Strategy: multiple greedy strategies with vectorized validity checks,
    local improvement passes, and a wall-clock deadline guard.
    """
    n = 8
    deadline = time.monotonic() + 595.0  # hard wall-clock limit with margin
    safety_margin = 10.0                 # stop searching this many seconds early

    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    grid_sq = (grid ** 2).sum(axis=1)  # precompute |p|^2 for the whole grid

    def can_add(arr: np.ndarray, arr_sq: np.ndarray, p: np.ndarray, p_sq: float) -> bool:
        """Check if adding point p to arr keeps the set sphere-free."""
        m = len(arr)
        if m < 4:
            return True
        
        # Lift each existing point to a 4-vector relative to candidate p
        w = np.empty((m, 4), dtype=np.float64)
        w[:, 0:3] = arr - p
        w[:, 3] = arr_sq - p_sq
        
        # Need to check all C(m,4) 4-tuples for linear dependence
        # Use vectorized approach: generate indices and compute dets in batches
        if m <= 64:
            # For small m, generate all combinations at once
            indices = np.array(list(itertools.combinations(range(m), 4)))
            if len(indices) == 0:
                return True
            # Flatten to get row indices for all matrices
            flat_indices = indices.reshape(-1)
            mats = w[flat_indices].reshape(-1, 4, 4)
            dets = np.linalg.det(mats)
            return not np.any(np.abs(dets) < 0.5)
        else:
            # For large m, process in chunks to manage memory
            chunk_size = 5000
            all_indices = list(itertools.combinations(range(m), 4))
            for start in range(0, len(all_indices), chunk_size):
                chunk = all_indices[start:start + chunk_size]
                if not chunk:
                    break
                flat = np.array([idx for combo in chunk for idx in combo])
                mats = w[flat].reshape(-1, 4, 4)
                dets = np.linalg.det(mats)
                if np.any(np.abs(dets) < 0.5):
                    return False
            return True

    def local_improve(arr: np.ndarray, arr_sq: np.ndarray, skipped: list, deadline: float, rng_local: np.random.Generator) -> tuple:
        """Try to insert previously skipped points into arr."""
        arr = arr.copy()
        arr_sq = arr_sq.copy()
        improved = True
        max_passes = 5
        for _ in range(max_passes):
            if not improved:
                break
            if time.monotonic() + safety_margin >= deadline:
                break
            improved = False
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
                else:
                    i += 1
        return arr, arr_sq, skipped

    def greedy_random(order: np.ndarray, deadline: float) -> tuple:
        """Greedy with given order."""
        arr = np.empty((0, 3), dtype=np.int64)
        arr_sq = np.empty(0, dtype=np.float64)
        skipped = []
        for p in order:
            if time.monotonic() + safety_margin >= deadline:
                break
            p_sq = float((p ** 2).sum())
            if can_add(arr, arr_sq, p, p_sq):
                arr = np.vstack([arr, p]) if len(arr) > 0 else p[None, :]
                arr_sq = np.append(arr_sq, p_sq)
            else:
                skipped.append((p, p_sq))
        return arr, arr_sq, skipped

    def greedy_by_density(order: np.ndarray, deadline: float) -> tuple:
        """Greedy preferring points that are far from existing ones."""
        arr = np.empty((0, 3), dtype=np.int64)
        arr_sq = np.empty(0, dtype=np.float64)
        remaining = list(order)
        skipped = []
        
        while remaining and time.monotonic() + safety_margin < deadline:
            if not arr.shape[0]:
                # First point: just take the first
                p = remaining.pop(0)
                p_sq = float((p ** 2).sum())
                arr = p[None, :]
                arr_sq = np.array([p_sq])
                continue
            
            # Find the point in remaining that maximizes min distance to arr
            best_idx = -1
            best_min_dist = -1
            # Sample if remaining is large
            if len(remaining) > 200:
                sample = np.random.default_rng(42).choice(len(remaining), size=200, replace=False)
            else:
                sample = list(range(len(remaining)))
            
            for idx in sample:
                p = remaining[idx]
                dists = np.linalg.norm(arr - p, axis=1)
                min_dist = dists.min()
                if min_dist > best_min_dist:
                    best_min_dist = min_dist
                    best_idx = idx
            
            if best_idx >= 0:
                p = remaining.pop(best_idx)
                p_sq = float((p ** 2).sum())
                if can_add(arr, arr_sq, p, p_sq):
                    arr = np.vstack([arr, p])
                    arr_sq = np.append(arr_sq, p_sq)
                else:
                    skipped.append((p, p_sq))
            else:
                break
        
        # Add remaining points in random order
        np.random.default_rng(42).shuffle(remaining)
        for p in remaining:
            if time.monotonic() + safety_margin >= deadline:
                break
            p_sq = float((p ** 2).sum())
            if can_add(arr, arr_sq, p, p_sq):
                arr = np.vstack([arr, p])
                arr_sq = np.append(arr_sq, p_sq)
            else:
                skipped.append((p, p_sq))
        
        return arr, arr_sq, skipped

    rng = np.random.default_rng(0)
    best = None
    best_len = 0
    
    # Strategy 1: Multiple random restarts
    for restart in range(200):
        if time.monotonic() + safety_margin >= deadline:
            break
        order = grid[rng.permutation(len(grid))]
        arr, arr_sq, skipped = greedy_random(order, deadline)
        arr, arr_sq, skipped = local_improve(arr, arr_sq, skipped, deadline, rng)
        
        if arr.shape[0] > best_len:
            best = arr
            best_len = arr.shape[0]
    
    # Strategy 2: Density-based greedy
    for restart in range(50):
        if time.monotonic() + safety_margin >= deadline:
            break
        order = grid[rng.permutation(len(grid))]
        arr, arr_sq, skipped = greedy_by_density(order, deadline)
        arr, arr_sq, skipped = local_improve(arr, arr_sq, skipped, deadline, rng)
        
        if arr.shape[0] > best_len:
            best = arr
            best_len = arr.shape[0]
    
    # Final polish: try to add any grid point not in best
    if best is not None and best_len > 0:
        in_best = set(map(tuple, best))
        candidates = [p for p in grid if tuple(p) not in in_best]
        rng.shuffle(candidates)
        for p in candidates:
            if time.monotonic() + safety_margin >= deadline:
                break
            p_sq = float((p ** 2).sum())
            if can_add(best, np.array([(pt**2).sum() for pt in best]), p, p_sq):
                best = np.vstack([best, p])
                best_len += 1
    
    if best is None:
        best = np.empty((0, 3), dtype=np.int64)

    return best.astype(np.int64)


# EVOLVE-BLOCK-END
