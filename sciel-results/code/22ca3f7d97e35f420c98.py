# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time


def no_5_on_a_sphere8() -> np.ndarray:
    """
    Maximize points in 8x8x8 grid with no 5 co-spherical/co-planar points.
    Uses randomized greedy + local search (swap moves) to escape local optima.
    """
    n = 8
    deadline = time.time() + 590  # safety margin before 630s kill

    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    N = len(grid)  # 512

    # Precompute squared norms
    sq_norms = (grid ** 2).sum(axis=1).astype(np.float64)

    def can_add_fast(arr: np.ndarray, p: np.ndarray) -> bool:
        """Check if adding point p to set arr keeps no-5-co-spherical property."""
        m = len(arr)
        if m < 4:
            return True
        
        # Compute lifted vectors: for each q in arr, row = (q - p, |q|^2 - |p|^2)
        diffs = arr - p  # (m, 3)
        norm_diff = sq_norms[:m] - (p ** 2).sum()  # (m,)
        w = np.column_stack([diffs, norm_diff])  # (m, 4)

        # Check all C(m,4) combinations
        # For m up to ~50, this is manageable
        if m <= 55:
            # Generate all 4-combinations
            idx = np.arange(m)
            combos = np.array(list(itertools.combinations(idx, 4)))
            sub = w[combos]  # (C, 4, 4)
            dets = np.linalg.det(sub)
            return not np.any(np.abs(dets) < 0.5)
        else:
            # Sample if too many
            rng = np.random.default_rng(99)
            sample_size = min(m, 35)
            sampled_idx = rng.choice(m, size=sample_size, replace=False)
            combos = np.array(list(itertools.combinations(sampled_idx, 4)))
            sub = w[combos]
            dets = np.linalg.det(sub)
            return not np.any(np.abs(dets) < 0.5)

    def greedy_build(order, start_set=None):
        """Greedy: add points in given order if valid."""
        if start_set is not None:
            arr = start_set.copy()
        else:
            arr = np.empty((0, 3), dtype=np.int64)
        for idx in order:
            p = grid[idx]
            if can_add_fast(arr, p):
                if len(arr) == 0:
                    arr = p[None, :]
                else:
                    arr = np.vstack([arr, p])
        return arr

    def local_search(arr, max_iters=200, rng=None):
        """Local search: try removing each point and greedily rebuilding."""
        if rng is None:
            rng = np.random.default_rng(42)
        best_arr = arr.copy()
        best_len = len(best_arr)
        
        for it in range(max_iters):
            if time.time() > deadline - 20:
                break
            improved = False
            # Try removing each point in random order
            order = rng.permutation(len(best_arr))
            for i in order:
                if time.time() > deadline - 20:
                    break
                # Remove point i
                reduced = np.delete(best_arr, i, axis=0)
                reduced_set = set(map(tuple, reduced.tolist()))
                
                # Try to add back points not in reduced, in random order
                candidates = [j for j in range(N) if tuple(grid[j]) not in reduced_set]
                rng.shuffle(candidates)
                
                new_arr = reduced.copy() if len(reduced) > 0 else np.empty((0, 3), dtype=np.int64)
                for j in candidates:
                    if time.time() > deadline - 20:
                        break
                    p = grid[j]
                    if can_add_fast(new_arr, p):
                        if len(new_arr) == 0:
                            new_arr = p[None, :]
                        else:
                            new_arr = np.vstack([new_arr, p])
                
                if len(new_arr) > best_len:
                    best_arr = new_arr
                    best_len = len(new_arr)
                    improved = True
                    break  # restart with new best
            
            if not improved:
                break
        
        return best_arr

    # Main search
    np.random.seed(0)
    best = None
    best_len = 0

    # Phase 1: Multiple greedy restarts
    num_greedy = 30
    for r in range(num_greedy):
        if time.time() > deadline - 90:
            break
        order = np.random.permutation(N)
        arr = greedy_build(order)
        if len(arr) > best_len:
            best_len = len(arr)
            best = arr.copy()

    # Phase 2: Local search on best
    if best is not None:
        improved = local_search(best, max_iters=300)
        if len(improved) > best_len:
            best = improved
            best_len = len(best)

        # Try local search on a few other greedy solutions
        for r in range(15):
            if time.time() > deadline - 30:
                break
            order = np.random.permutation(N)
            arr = greedy_build(order)
            if len(arr) >= best_len - 3:
                improved = local_search(arr, max_iters=150)
                if len(improved) > best_len:
                    best = improved
                    best_len = len(best)

    # Phase 3: Final polish
    if best is not None and time.time() < deadline - 30:
        improved = local_search(best, max_iters=500)
        if len(improved) > best_len:
            best = improved
            best_len = len(best)

    if best is None:
        return np.array([], dtype=np.int64).reshape(0, 3)

    return best.astype(np.int64)


# EVOLVE-BLOCK-END
