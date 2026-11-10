# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time
from math import comb


def no_5_on_a_sphere8() -> np.ndarray:
    n = 8
    start_time = time.time()
    deadline = start_time + 620.0  # safety margin before 630s kill

    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    grid_sq = (grid ** 2).sum(axis=1).astype(np.float64)

    def can_add_vectorized(arr: np.ndarray, p: np.ndarray, arr_sq: np.ndarray) -> bool:
        """Check if p can be added to arr without creating 5 co-spherical points.
        Uses vectorized determinant computation for speed."""
        m = len(arr)
        if m < 4:
            return True

        # Build lifted matrix: rows = (q_i - p, |q_i|^2 - |p|^2)
        w = np.empty((m, 4), dtype=np.float64)
        w[:, 0:3] = arr.astype(np.float64) - p.astype(np.float64)
        w[:, 3] = arr_sq - (p.astype(np.float64) ** 2).sum()

        # Check all C(m,4) combinations; if any 4x4 det is ~0, reject
        # For small m, use itertools; for larger m, use vectorized approach
        if m <= 8:
            for combo in itertools.combinations(range(m), 4):
                sub = w[list(combo)]
                d = np.linalg.det(sub)
                if abs(d) < 0.5:
                    return False
            return True

        # Vectorized approach: compute all C(m,4) determinants at once
        # This is memory intensive but much faster for moderate m
        # Generate all index combinations
        indices = list(itertools.combinations(range(m), 4))
        if len(indices) == 0:
            return True

        # Stack all 4x4 submatrices
        idx_arr = np.array(indices, dtype=np.int64)  # shape (num_combos, 4)
        # Gather rows: w[idx_arr[i, j]] for i in range(num_combos), j in range(4)
        # Shape: (num_combos, 4, 4)
        submatrices = w[idx_arr]  # fancy indexing gives (num_combos, 4, 4)

        # Compute determinants of all 4x4 matrices at once
        # np.linalg.det works on stacks of matrices
        dets = np.linalg.det(submatrices)

        # Check if any determinant is too small
        if np.any(np.abs(dets) < 0.5):
            return False
        return True

    def greedy_fill(order: np.ndarray, arr: np.ndarray = None, arr_sq: np.ndarray = None) -> np.ndarray:
        """Greedy fill using the given order. Returns the resulting array."""
        if arr is None:
            arr = np.empty((0, 3), dtype=np.int64)
            arr_sq = np.empty(0, dtype=np.float64)

        for idx in order:
            if time.time() > deadline - 10:
                break
            p = grid[idx]
            if can_add_vectorized(arr, p, arr_sq):
                if len(arr) == 0:
                    arr = p[None, :]
                    arr_sq = np.array([grid_sq[idx]], dtype=np.float64)
                else:
                    arr = np.vstack([arr, p])
                    arr_sq = np.append(arr_sq, grid_sq[idx])
        return arr

    def local_search(arr: np.ndarray, max_iters: int = 500) -> np.ndarray:
        """Try to improve the set by swapping points.
        For each point in the set, try replacing it with a point not in the set."""
        if len(arr) == 0:
            return arr

        best_arr = arr.copy()
        best_len = len(arr)
        arr_sq = (arr.astype(np.float64) ** 2).sum(axis=1)

        in_set = set(map(tuple, arr.tolist()))
        all_indices = set(range(len(grid)))
        in_set_indices = set(grid.tolist())
        candidate_indices = [i for i in range(len(grid)) if tuple(grid[i]) not in in_set]

        rng = np.random.default_rng(12345)
        rng.shuffle(candidate_indices)

        improved = True
        while improved and time.time() < deadline - 20:
            improved = False
            for old_idx in range(len(best_arr)):
                if time.time() > deadline - 20:
                    break
                old_point = best_arr[old_idx]
                old_sq = (old_point.astype(np.float64) ** 2).sum()

                # Try replacing old_point with each candidate
                for new_idx in candidate_indices:
                    if time.time() > deadline - 20:
                        break
                    new_point = grid[new_idx]
                    new_sq = grid_sq[new_idx]

                    # Build new array
                    new_arr = best_arr.copy()
                    new_arr[old_idx] = new_point
                    new_arr_sq = (new_arr.astype(np.float64) ** 2).sum(axis=1)

                    # Check validity
                    valid = True
                    m = len(new_arr)
                    if m >= 4:
                        w = np.empty((m, 4), dtype=np.float64)
                        w[:, 0:3] = new_arr.astype(np.float64)
                        w[:, 3] = new_arr_sq
                        # Check all C(m,4) determinants
                        for combo in itertools.combinations(range(m), 4):
                            sub = w[list(combo)]
                            d = np.linalg.det(sub)
                            if abs(d) < 0.5:
                                valid = False
                                break

                    if valid:
                        best_arr = new_arr
                        best_len = len(best_arr)
                        in_set = set(map(tuple, best_arr.tolist()))
                        improved = True
                        # Remove the new point from candidates
                        candidate_indices = [i for i in candidate_indices if i != new_idx]
                        break

        return best_arr

    def greedy_with_priority(order: np.ndarray) -> np.ndarray:
        """Greedy fill with priority: at each step, choose the point that is most 'compatible'
        (has the largest minimum determinant when added)."""
        arr = np.empty((0, 3), dtype=np.int64)
        arr_sq = np.empty(0, dtype=np.float64)
        remaining = list(order)

        while remaining and time.time() < deadline - 15:
            if not remaining:
                break

            # If we have fewer than 4 points, just take the first one
            if len(arr) < 4:
                idx = remaining.pop(0)
                p = grid[idx]
                if len(arr) == 0:
                    arr = p[None, :]
                    arr_sq = np.array([grid_sq[idx]], dtype=np.float64)
                else:
                    arr = np.vstack([arr, p])
                    arr_sq = np.append(arr_sq, grid_sq[idx])
                continue

            # For each candidate, compute the minimum |det| when added
            # Choose the one with the largest minimum |det|
            best_idx = -1
            best_min_det = -1
            best_p = None
            best_sq = None

            # Limit the number of candidates we evaluate per step for speed
            eval_count = min(len(remaining), 20)
            for i in range(eval_count):
                idx = remaining[i]
                p = grid[idx]
                p_sq = grid_sq[idx]

                # Build lifted matrix with existing points
                m = len(arr)
                w = np.empty((m, 4), dtype=np.float64)
                w[:, 0:3] = arr.astype(np.float64) - p.astype(np.float64)
                w[:, 3] = arr_sq - p_sq

                # Compute all C(m,4) determinants
                min_det = np.inf
                for combo in itertools.combinations(range(m), 4):
                    sub = w[list(combo)]
                    d = np.linalg.det(sub)
                    abs_d = abs(d)
                    if abs_d < min_det:
                        min_det = abs_d
                    if abs_d < 0.5:
                        break

                if min_det > best_min_det and min_det >= 0.5:
                    best_min_det = min_det
                    best_idx = i
                    best_p = p
                    best_sq = p_sq

            if best_idx >= 0 and best_p is not None:
                remaining.pop(best_idx)
                arr = np.vstack([arr, best_p])
                arr_sq = np.append(arr_sq, best_sq)
            else:
                # No valid point found in the first eval_count candidates
                # Try the rest
                found = False
                for i in range(eval_count, len(remaining)):
                    idx = remaining[i]
                    p = grid[idx]
                    if can_add_vectorized(arr, p, arr_sq):
                        remaining.pop(i)
                        arr = np.vstack([arr, p])
                        arr_sq = np.append(arr_sq, grid_sq[idx])
                        found = True
                        break
                if not found:
                    break

        return arr

    best = np.empty((0, 3), dtype=np.int64)
    best_len = 0

    rng = np.random.default_rng(42)
    num_restarts = 200  # Increased restarts with faster checks

    for restart in range(num_restarts):
        if time.time() > deadline - 30:  # 30s safety margin
            break

        # Vary the ordering strategy
        strategy = restart % 5
        if strategy == 0:
            # Pure random
            order = rng.permutation(len(grid))
        elif strategy == 1:
            # Bias toward corners/edges
            dist_from_center = np.sum((grid - 3.5) ** 2, axis=1)
            weights = dist_from_center + 1e-9
            weights /= weights.sum()
            order = rng.choice(len(grid), size=len(grid), replace=False, p=weights)
        elif strategy == 2:
            # Bias toward center
            dist_from_center = np.sum((grid - 3.5) ** 2, axis=1)
            max_dist = dist_from_center.max()
            weights = (max_dist + 1.0 - dist_from_center) + 1e-9
            weights /= weights.sum()
            order = rng.choice(len(grid), size=len(grid), replace=False, p=weights)
        elif strategy == 3:
            # Random with slight bias toward high coordinates
            coord_sum = grid.sum(axis=1)
            weights = coord_sum + 1e-9
            weights /= weights.sum()
            order = rng.choice(len(grid), size=len(grid), replace=False, p=weights)
        else:
            # Random with slight bias toward low coordinates
            coord_sum = grid.sum(axis=1)
            weights = (14 - coord_sum) + 1e-9
            weights /= weights.sum()
            order = rng.choice(len(grid), size=len(grid), replace=False, p=weights)

        # Use priority-based greedy for some restarts
        if restart % 3 == 0:
            result = greedy_with_priority(order)
        else:
            result = greedy_fill(order)

        if len(result) > best_len:
            best = result.copy()
            best_len = len(result)

            # Try local search on the best result
            if len(result) >= 5 and time.time() < deadline - 60:
                polished = local_search(result)
                if len(polished) > best_len:
                    best = polished.copy()
                    best_len = len(polished)

    return best.astype(np.int64)


# EVOLVE-BLOCK-END
