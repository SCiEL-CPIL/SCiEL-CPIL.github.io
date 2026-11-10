# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time


def no_5_on_a_sphere8() -> np.ndarray:
    """
    The "no 5 on a sphere" problem. Select as many points as possible from the
    8x8x8 integer grid such that no five of the selected points are co-spherical
    (or co-planar).

    Uses randomized greedy search with multiple restarts, batch determinant
    checks, pruning, and a local improvement phase with simulated annealing.
    """
    n = 8
    deadline = time.time() + 610  # Safety margin before 630s limit

    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    num_points = len(grid)
    sq_norms_grid = (grid ** 2).sum(axis=1)

    def can_add(arr: np.ndarray, p: np.ndarray, sq_p: float = None) -> bool:
        m = len(arr)
        if m < 4:
            return True
        if sq_p is None:
            sq_p = (p ** 2).sum()
        
        diff = arr - p
        sq_diff = (arr ** 2).sum(axis=1) - sq_p
        W = np.column_stack((diff, sq_diff))
        
        if m == 4:
            det = np.linalg.det(W)
            return abs(det) >= 0.5
        
        combos = list(itertools.combinations(range(m), 4))
        sub = W[combos]  # (C, 4, 4)
        dets = np.linalg.det(sub)
        return not np.any(np.abs(dets) < 0.5)

    def local_improve(arr: np.ndarray, rng: np.random.Generator, time_limit: float) -> np.ndarray:
        """Try to improve the set by removing one point and adding two others."""
        best_arr = arr.copy()
        best_len = len(best_arr)
        
        while time.time() < time_limit:
            improved = False
            m = len(best_arr)
            if m < 3:
                break
            
            # Try removing each point and see if we can add 2 more
            for i in range(m):
                if time.time() >= time_limit:
                    break
                trial = np.delete(best_arr, i, axis=0)
                removed_p = best_arr[i]
                
                # Try to add points from the remaining grid
                candidates = [j for j in range(num_points) if not np.any((grid[j] == removed_p).all())]
                rng.shuffle(candidates)
                
                added = 0
                for j in candidates:
                    if time.time() >= time_limit:
                        break
                    if added >= 2:
                        break
                    p = grid[j]
                    if can_add(trial, p, sq_norms_grid[j]):
                        trial = np.vstack([trial, p]) if len(trial) > 0 else p[None, :]
                        added += 1
                
                if len(trial) > best_len:
                    best_arr = trial
                    best_len = len(trial)
                    improved = True
            
            if not improved:
                break
        
        return best_arr

    rng = np.random.default_rng(42)
    best = None
    best_len = 0

    restart = 0
    while time.time() < deadline - 30:
        restart += 1
        
        # Random greedy pass
        order = rng.permutation(num_points)
        arr = np.empty((0, 3), dtype=np.int64)
        
        for i, idx in enumerate(order):
            # Pruning: if current size + remaining <= best size, break
            if len(arr) + (num_points - i) <= best_len:
                break
            
            p = grid[idx]
            if can_add(arr, p, sq_norms_grid[idx]):
                arr = np.vstack([arr, p]) if len(arr) > 0 else p[None, :]
        
        # Local improvement phase
        remaining_time = deadline - 30 - time.time()
        if remaining_time > 5 and len(arr) >= best_len:
            arr = local_improve(arr, rng, time.time() + min(remaining_time * 0.3, 30))
        
        if best is None or len(arr) > best_len:
            best = arr.copy()
            best_len = len(arr)
        
        # If we've found a very large set, we can be more conservative
        if best_len >= 40 and time.time() > deadline - 120:
            break

    return best.astype(np.int64) if best is not None else np.empty((0, 3), dtype=np.int64)


# EVOLVE-BLOCK-END
