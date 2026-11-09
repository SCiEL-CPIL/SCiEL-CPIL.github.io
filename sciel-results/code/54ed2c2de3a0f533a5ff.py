# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def no_5_on_a_sphere8() -> np.ndarray:
    """
    Maximize points in 8x8x8 grid with no 5 co-spherical.
    Uses Max-Min Distance Greedy + Iterated Local Search (ILS).
    """
    n = 8
    deadline = time.time() + 590  # 10s safety margin
    
    # All grid points
    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    N = len(grid)  # 512
    sq = (grid ** 2).sum(axis=1)  # |p|^2 for each grid point

    def is_valid_set(indices: np.ndarray) -> bool:
        """Check if the set of grid indices has no 5 co-spherical points."""
        m = len(indices)
        if m < 5:
            return True
        pts = grid[indices]
        sqs = sq[indices]
        for i in range(m):
            p = pts[i]
            pi_sq = sqs[i]
            others = np.delete(indices, i)
            if len(others) < 4:
                continue
            # Check all combinations of 4 others
            # To speed up, we can break early
            for combo in itertools.combinations(range(len(others)), 4):
                q1, q2, q3, q4 = others[list(combo)]
                # Form 4x4 matrix
                # Rows: (q - p, |q|^2 - |p|^2)
                # We can construct the matrix directly
                m_mat = np.empty((4, 4), dtype=np.float64)
                m_mat[0, 0:3] = grid[q1] - p
                m_mat[0, 3] = sq[q1] - pi_sq
                m_mat[1, 0:3] = grid[q2] - p
                m_mat[1, 3] = sq[q2] - pi_sq
                m_mat[2, 0:3] = grid[q3] - p
                m_mat[2, 3] = sq[q3] - pi_sq
                m_mat[3, 0:3] = grid[q4] - p
                m_mat[3, 3] = sq[q4] - pi_sq
                
                if abs(np.linalg.det(m_mat)) < 0.5:
                    return False
        return True

    def can_add_to_set(arr_indices: np.ndarray, new_idx: int) -> bool:
        """Check if adding new_idx to the set keeps it valid."""
        m = len(arr_indices)
        if m < 4:
            return True
        p = grid[new_idx]
        pi_sq = sq[new_idx]
        others = arr_indices
        # Check all combinations of 4 from 'others'
        for combo in itertools.combinations(range(m), 4):
            q1, q2, q3, q4 = others[list(combo)]
            m_mat = np.empty((4, 4), dtype=np.float64)
            m_mat[0, 0:3] = grid[q1] - p
            m_mat[0, 3] = sq[q1] - pi_sq
            m_mat[1, 0:3] = grid[q2] - p
            m_mat[1, 3] = sq[q2] - pi_sq
            m_mat[2, 0:3] = grid[q3] - p
            m_mat[2, 3] = sq[q3] - pi_sq
            m_mat[3, 0:3] = grid[q4] - p
            m_mat[3, 3] = sq[q4] - pi_sq
            
            if abs(np.linalg.det(m_mat)) < 0.5:
                return False
        return True

    def greedy_build_maxmin() -> np.ndarray:
        """Greedy: add points that maximize minimum distance to existing set."""
        arr = np.empty(0, dtype=np.int64)
        arr_set = set()
        remaining = np.arange(N)
        
        # Start with a random point
        first_idx = np.random.choice(N)
        arr = np.array([first_idx], dtype=np.int64)
        arr_set.add(first_idx)
        remaining = np.delete(remaining, np.where(remaining == first_idx)[0])
        
        while len(remaining) > 0 and time.time() < deadline:
            # Compute min distance from each remaining point to arr
            # arr shape (k, 3), remaining shape (m, 3)
            # dist[i, j] = ||remaining[i] - arr[j]||^2
            if len(arr) == 0:
                min_dists = np.zeros(len(remaining))
            else:
                # Vectorized distance calculation
                # remaining[:, np.newaxis, :] - arr[np.newaxis, :, :]
                diffs = remaining[:, np.newaxis, :] - arr[np.newaxis, :, :]
                dists = np.sum(diffs ** 2, axis=2)
                min_dists = np.min(dists, axis=1)
            
            # Pick the point with max min_dist
            best_local_idx = np.argmax(min_dists)
            candidate_idx = remaining[best_local_idx]
            
            if can_add_to_set(arr, candidate_idx):
                arr = np.append(arr, candidate_idx)
                arr_set.add(candidate_idx)
                remaining = np.delete(remaining, best_local_idx)
            else:
                # If invalid, remove it from remaining and try next best
                remaining = np.delete(remaining, best_local_idx)
                # To avoid infinite loop if all remaining are invalid, 
                # we just continue. The loop condition handles it.
                # However, if we remove all, we stop.
                # We should probably just stop if we can't add any more?
                # No, we can skip this one and try the next best.
                # But the "next best" is now the max of the remaining min_dists.
                # We need to recompute min_dists or just iterate.
                # For efficiency, we recompute in the next iteration.
                pass
        
        return arr

    def local_search_1swap(arr: np.ndarray) -> np.ndarray:
        """Try swap moves: remove one, add another."""
        current = arr.copy()
        current_set = set(current.tolist())
        all_indices = np.arange(N)
        comp = np.array([i for i in all_indices if i not in current_set], dtype=np.int64)
        
        improved = True
        swaps_done = 0
        max_swaps = 100
        while improved and swaps_done < max_swaps and time.time() < deadline:
            improved = False
            for i in range(len(current)):
                if time.time() > deadline:
                    break
                removed = current[i]
                without = np.delete(current, i)
                for j in range(len(comp)):
                    if time.time() > deadline:
                        break
                    add_idx = comp[j]
                    if can_add_to_set(without, add_idx):
                        new_arr = np.concatenate([without, [add_idx]])
                        current = new_arr
                        current_set = set(current.tolist())
                        comp = np.array([i for i in all_indices if i not in current_set], dtype=np.int64)
                        improved = True
                        swaps_done += 1
                        break
                if improved:
                    break
        return current

    def local_search_add(arr: np.ndarray) -> np.ndarray:
        """Try to add more points."""
        current = arr.copy()
        current_set = set(current.tolist())
        all_indices = np.arange(N)
        comp = np.array([i for i in all_indices if i not in current_set], dtype=np.int64)
        
        for idx in comp:
            if time.time() > deadline:
                break
            if can_add_to_set(current, idx):
                current = np.append(current, idx)
                current_set.add(idx)
                comp = np.array([i for i in all_indices if i not in current_set], dtype=np.int64)
        return current

    rng = np.random.RandomState(42)
    best = None
    best_len = 0
    
    # Phase 1: Multiple Greedy Max-Min restarts
    # We run this until time runs out or we find a very good solution
    while time.time() < deadline - 60:
        arr = greedy_build_maxmin()
        if len(arr) > best_len:
            best = arr
            best_len = len(arr)
        
        # Local search on this result
        if len(arr) > 0:
            arr_ls = local_search_1swap(arr)
            arr_ls = local_search_add(arr_ls)
            if len(arr_ls) > best_len:
                best = arr_ls
                best_len = len(arr_ls)

    # Phase 2: ILS on the best solution found
    if best is not None and time.time() < deadline:
        current_best = best
        current_best_len = best_len
        while time.time() < deadline - 10:
            # Perturb: remove 2 random points, add 2 random others
            if len(current_best) < 5:
                break
            perturbed = current_best.copy()
            # Remove 2
            if len(perturbed) >= 2:
                to_remove = rng.choice(len(perturbed), size=2, replace=False)
                to_remove.sort()
                mask = np.ones(len(perturbed), dtype=bool)
                mask[to_remove[0]] = False
                mask[to_remove[1]] = False
                perturbed = perturbed[mask]
            
            # Add 2 random from complement
            current_set = set(perturbed.tolist())
            all_indices = np.arange(N)
            comp = np.array([i for i in all_indices if i not in current_set], dtype=np.int64)
            if len(comp) >= 2:
                to_add = rng.choice(len(comp), size=2, replace=False)
                for idx in to_add:
                    if can_add_to_set(perturbed, comp[idx]):
                        perturbed = np.append(perturbed, comp[idx])
                        current_set.add(comp[idx])
            
            # Local search
            perturbed = local_search_1swap(perturbed)
            perturbed = local_search_add(perturbed)
            
            if len(perturbed) > current_best_len:
                current_best = perturbed
                current_best_len = len(perturbed)
                best = current_best
                best_len = current_best_len

    if best is None:
        best = np.array([0, 1, 2, 3], dtype=np.int64)
    
    return grid[best].astype(np.int64)

# EVOLVE-BLOCK-END
