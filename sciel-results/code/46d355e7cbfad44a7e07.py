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

    Strategy: Greedy construction with multiple random restarts, followed by
    local search via swap moves to escape greedy local optima.

    Returns:
        points: np.ndarray of shape (k, 3) of integer grid coordinates in
            [0, n-1], no five of which are co-spherical.
    """
    n = 8
    deadline = time.time() + 580.0  # safety margin before 630s kill

    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    grid_sq = (grid ** 2).sum(axis=1).astype(np.float64)
    N = len(grid)
    # Map from (x,y,z) to index
    idx_map = {}
    for idx, (x, y, z) in enumerate(grid):
        idx_map[(x, y, z)] = idx

    def can_add_to_set(arr: np.ndarray, arr_sq: np.ndarray, p: np.ndarray, p_sq: float) -> bool:
        """Check if adding point p to set arr keeps no-5-spherical."""
        m = len(arr)
        if m < 4:
            return True
        w = np.empty((m, 4), dtype=np.float64)
        w[:, 0:3] = arr - p
        w[:, 3] = arr_sq - p_sq
        combos = list(itertools.combinations(range(m), 4))
        if not combos:
            return True
        sub = w[np.array(combos)]
        dets = np.linalg.det(sub)
        return not np.any(np.abs(dets) < 0.5)

    def greedy_build(order: np.ndarray) -> tuple:
        """Greedy: add points in given order if valid."""
        arr = np.empty((0, 3), dtype=np.int64)
        arr_sq = np.empty((0,), dtype=np.float64)
        for idx in order:
            p = grid[idx]
            p_sq = grid_sq[idx]
            if can_add_to_set(arr, arr_sq, p, p_sq):
                if len(arr) == 0:
                    arr = p[None, :]
                    arr_sq = np.array([p_sq])
                else:
                    arr = np.vstack([arr, p[None, :]])
                    arr_sq = np.append(arr_sq, p_sq)
        return arr, arr_sq

    def local_search(arr: np.ndarray, arr_sq: np.ndarray, budget: float) -> tuple:
        """Local search via swap moves."""
        arr = arr.copy()
        arr_sq = arr_sq.copy()
        in_set = np.zeros(N, dtype=bool)
        for i in range(len(arr)):
            x, y, z = arr[i]
            in_set[idx_map[(x, y, z)]] = True

        improved = True
        while improved and time.time() < budget:
            improved = False
            k = len(arr)
            for i in range(k):
                if time.time() >= budget:
                    break
                removed = arr[i]
                removed_idx = idx_map[tuple(removed)]
                
                if k > 1:
                    red_arr = np.delete(arr, i, axis=0)
                    red_sq = np.delete(arr_sq, i)
                else:
                    red_arr = np.empty((0, 3), dtype=np.int64)
                    red_sq = np.empty((0,), dtype=np.float64)

                # Get candidate indices not in set
                cand_mask = ~in_set
                cand_mask[removed_idx] = True  # removed point is now available
                candidates = np.where(cand_mask)[0]
                if len(candidates) == 0:
                    continue
                np.random.shuffle(candidates)

                # Try candidates, limit to first 50 for speed
                for j in candidates[:50]:
                    if time.time() >= budget:
                        break
                    p = grid[j]
                    p_sq = grid_sq[j]
                    if can_add_to_set(red_arr, red_sq, p, p_sq):
                        # Accept swap
                        if k > 1:
                            arr = np.vstack([red_arr, p[None, :]])
                            arr_sq = np.append(red_sq, p_sq)
                        else:
                            arr = p[None, :]
                            arr_sq = np.array([p_sq])
                        in_set[removed_idx] = False
                        in_set[j] = True
                        improved = True
                        break

        return arr, arr_sq

    np.random.seed(0)
    best_arr = None
    best_sq = None
    best_len = 0

    num_restarts = 30
    for r in range(num_restarts):
        if time.time() >= deadline - 60.0:
            break

        order = np.random.permutation(N)
        arr, arr_sq = greedy_build(order)

        # Local search with time budget
        ls_budget = min(deadline - 30.0, time.time() + 20.0)
        arr, arr_sq = local_search(arr, arr_sq, ls_budget)

        if len(arr) > best_len:
            best_len = len(arr)
            best_arr = arr.copy()
            best_sq = arr_sq.copy()

    if best_arr is None:
        order = np.random.permutation(N)
        best_arr, best_sq = greedy_build(order)

    return best_arr.astype(np.int64)


# EVOLVE-BLOCK-END
