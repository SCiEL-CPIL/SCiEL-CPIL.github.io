# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time


def no_5_on_a_sphere8() -> np.ndarray:
    n = 8
    deadline = time.time() + 600.0
    safety_margin = 30.0  # seconds before deadline to stop

    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    n_points_total = len(grid)

    # Precompute squared norms for all grid points
    grid_sq_norms = (grid ** 2).sum(axis=1).astype(np.float64)

    def can_add(arr: np.ndarray, p: np.ndarray, arr_sq_norms: np.ndarray) -> bool:
        """Check if adding point p to arr keeps the set valid (no 5 co-spherical)."""
        m = len(arr)
        if m < 4:
            return True

        p_sq = float((p ** 2).sum())
        diffs = arr.astype(np.float64) - p.astype(np.float64)
        sq_diffs = arr_sq_norms - p_sq
        rows = np.column_stack([diffs, sq_diffs])  # shape (m, 4)

        # Use batched determinant for all C(m,4) combos at once when feasible
        if m <= 35:
            combos = np.array(list(itertools.combinations(range(m), 4)))
            sub = rows[combos]  # (C, 4, 4)
            dets = np.linalg.det(sub)
            return not np.any(np.abs(dets) < 0.5)
        else:
            # Incremental with early termination
            for combo in itertools.combinations(range(m), 4):
                sub = rows[list(combo)]
                det = np.linalg.det(sub)
                if abs(det) < 0.5:
                    return False
            return True

    def greedy_fill(order: np.ndarray, max_points: int = None) -> np.ndarray:
        """Greedy fill: add points in given order if valid."""
        arr = np.empty((0, 3), dtype=np.int64)
        arr_sq = np.empty(0, dtype=np.float64)
        arr_set = set()

        for idx in order:
            p = grid[idx]
            if can_add(arr, p, arr_sq):
                if len(arr) == 0:
                    arr = p[None, :]
                    arr_sq = np.array([grid_sq_norms[idx]])
                    arr_set.add(tuple(p))
                else:
                    arr = np.vstack([arr, p])
                    arr_sq = np.append(arr_sq, grid_sq_norms[idx])
                    arr_set.add(tuple(p))
                if max_points is not None and len(arr) >= max_points:
                    break

        return arr

    def local_search(arr: np.ndarray, max_iters: int = 100) -> np.ndarray:
        """Try to improve by swapping points: remove one, try adding another."""
        if len(arr) < 4:
            return arr

        arr_set = set(map(tuple, arr))
        current_arr = arr.copy()
        current_sq = (current_arr ** 2).sum(axis=1).astype(np.float64)
        improved = True
        iters = 0

        while improved and iters < max_iters and time.time() < deadline - safety_margin:
            improved = False
            iters += 1
            for i in range(len(current_arr)):
                # Try removing point i
                remaining = np.delete(current_arr, i, axis=0)
                remaining_sq = np.delete(current_sq, i, axis=0)

                # Try adding each grid point not already in set
                for j in range(n_points_total):
                    if j in arr_set:
                        continue
                    p = grid[j]
                    if can_add(remaining, p, remaining_sq):
                        new_arr = np.vstack([remaining, p])
                        new_sq = np.append(remaining_sq, grid_sq_norms[j])
                        new_set = set(map(tuple, new_arr))
                        if new_set != arr_set:
                            current_arr = new_arr
                            current_sq = new_sq
                            arr_set = new_set
                            improved = True
                            break
                if improved:
                    break
        return current_arr

    # Main search: multiple randomized greedy passes
    np.random.seed(42)
    best = None
    best_len = 0

    # Strategy 1: Pure random order greedy
    for restart in range(20):
        if time.time() > deadline - safety_margin:
            break
        order = np.random.permutation(n_points_total)
        arr = greedy_fill(order)
        if len(arr) > best_len:
            best = arr
            best_len = len(arr)

    # Strategy 2: Local search on best result
    if best is not None and time.time() < deadline - safety_margin:
        improved = local_search(best, max_iters=80)
        if len(improved) > best_len:
            best = improved
            best_len = len(improved)

    # Strategy 3: More greedy passes with different seeds
    for restart in range(50):
        if time.time() > deadline - safety_margin:
            break
        seed = 1000 + restart
        np.random.seed(seed)
        order = np.random.permutation(n_points_total)
        arr = greedy_fill(order)
        if len(arr) > best_len:
            best = arr
            best_len = len(arr)
        # Quick local search on promising results
        if len(arr) >= best_len - 1 and time.time() < deadline - safety_margin:
            improved = local_search(arr, max_iters=30)
            if len(improved) > best_len:
                best = improved
                best_len = len(improved)

    # Final local search on best
    if best is not None and time.time() < deadline - safety_margin:
        final = local_search(best, max_iters=150)
        if len(final) > best_len:
            best = final
            best_len = len(final)

    if best is None:
        # Fallback: just take first few points
        best = grid[:5]

    return best.astype(np.int64)


# EVOLVE-BLOCK-END
