# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time


def no_5_on_a_sphere8() -> np.ndarray:
    n = 8
    deadline = time.time() + 600.0  # safety margin before 630s kill

    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    grid_sq = (grid ** 2).sum(axis=1)  # |p|^2 for each grid point

    def can_add(arr: np.ndarray, p: np.ndarray, arr_sq: np.ndarray):
        """
        Returns (True, None) if p can be added.
        Returns (False, conflict_indices) if p cannot be added, where conflict_indices
        are the indices in arr that form a co-spherical set with p.
        """
        m = len(arr)
        if m < 4:
            return True, None
        # Build lifted matrix: rows = (q_i - p, |q_i|^2 - |p|^2)
        w = np.empty((m, 4), dtype=np.float64)
        w[:, 0:3] = arr - p
        w[:, 3] = arr_sq - (p ** 2).sum()
        
        # Check all C(m,4) combinations of 4 rows; if any 4x4 det is ~0, reject
        for combo in itertools.combinations(range(m), 4):
            sub = w[list(combo)]
            d = np.linalg.det(sub)
            if abs(d) < 0.5:
                return False, list(combo)
        return True, None

    def greedy_fill(order: np.ndarray) -> np.ndarray:
        arr = np.empty((0, 3), dtype=np.int64)
        arr_sq = np.empty(0, dtype=np.float64)
        for idx in order:
            if time.time() > deadline - 5:
                break
            p = grid[idx]
            ok, _ = can_add(arr, p, arr_sq)
            if ok:
                if len(arr) == 0:
                    arr = p[None, :]
                    arr_sq = np.array([grid_sq[idx]], dtype=np.float64)
                else:
                    arr = np.vstack([arr, p])
                    arr_sq = np.append(arr_sq, grid_sq[idx])
        return arr

    def local_search(arr: np.ndarray, arr_sq: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """
        Attempts to improve the set by adding points or swapping points.
        """
        max_steps = 200
        for _ in range(max_steps):
            if time.time() > deadline - 10:
                break
            
            # Try to add a random point
            cand_idx = rng.integers(len(grid))
            p = grid[cand_idx]
            
            # Check if p is already in arr
            if np.any(np.all(arr == p, axis=1)):
                continue
                
            ok, conflict = can_add(arr, p, arr_sq)
            
            if ok:
                # Add it
                arr = np.vstack([arr, p])
                arr_sq = np.append(arr_sq, grid_sq[cand_idx])
                continue
            
            # If not ok, try a swap
            # conflict is a list of 4 indices in arr
            # Pick one to remove
            remove_idx_in_arr = conflict[rng.integers(4)]
            
            # Create arr without the removed point
            mask = np.ones(len(arr), dtype=bool)
            mask[remove_idx_in_arr] = False
            arr_wo_q = arr[mask]
            arr_sq_wo_q = arr_sq[mask]
            
            # Check if p can be added to arr_wo_q
            ok_swap, _ = can_add(arr_wo_q, p, arr_sq_wo_q)
            
            if ok_swap:
                # Accept swap
                arr = np.vstack([arr_wo_q, p])
                arr_sq = np.append(arr_sq_wo_q, grid_sq[cand_idx])
                
        return arr

    best = np.empty((0, 3), dtype=np.int64)
    best_len = 0

    rng = np.random.default_rng(42)
    num_restarts = 50 # Reduced to allow time for local search

    for restart in range(num_restarts):
        if time.time() > deadline - 60:  # 60s safety margin for last few restarts
            break

        # Vary the ordering strategy
        if restart % 3 == 0:
            order = rng.permutation(len(grid))
        elif restart % 3 == 1:
            dist_from_center = np.sum((grid - 3.5) ** 2, axis=1)
            weights = dist_from_center + 1e-9
            weights /= weights.sum()
            order = rng.choice(len(grid), size=len(grid), replace=False, p=weights)
        else:
            dist_from_center = np.sum((grid - 3.5) ** 2, axis=1)
            max_dist = dist_from_center.max()
            weights = (max_dist + 1.0 - dist_from_center) + 1e-9
            weights /= weights.sum()
            order = rng.choice(len(grid), size=len(grid), replace=False, p=weights)

        result = greedy_fill(order)
        
        # Run local search on the result
        if len(result) > 0:
            # Reconstruct arr_sq for local search
            result_sq = (result ** 2).sum(axis=1)
            improved_arr = local_search(result, result_sq, rng)
            result = improved_arr

        if len(result) > best_len:
            best = result.copy()
            best_len = len(result)

    return best.astype(np.int64)


# EVOLVE-BLOCK-END
