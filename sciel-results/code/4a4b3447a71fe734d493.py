# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def _is_valid(points):
    """Check the full set for validity using the task's test definition."""
    if points is None or len(points) == 0:
        return True
    points = np.asarray(points, dtype=np.float64)
    n = len(points)
    if n < 5:
        return True
    for i in range(4, n):
        p = points[i]
        pn2 = np.dot(p, p)
        # For each set of 4 earlier points, build the 4x4 matrix and check det
        for combo in itertools.combinations(range(i), 4):
            qs = points[list(combo)]
            diffs = qs - p
            norms = np.sum(qs ** 2, axis=1) - pn2
            M = np.hstack((diffs, norms[:, None]))
            if abs(np.linalg.det(M)) < 0.5:
                return False
    return True

def _check_add(points, p, sq_norms_points, sq_norm_p):
    """Check if p can be added to the set of points."""
    m = len(points)
    if m < 4:
        return True
    diffs = points - p
    W = np.hstack((diffs, (sq_norms_points - sq_norm_p)[:, np.newaxis]))
    indices = np.array(list(itertools.combinations(range(m), 4)))
    if len(indices) == 0:
        return True
    sub_matrices = W[indices]
    dets = np.linalg.det(sub_matrices)
    return not np.any(np.abs(dets) < 0.5)

def _greedy_pass(seed, grid, sq_norms_all, deadline):
    """Run a single greedy pass with a specific random seed."""
    rng = np.random.RandomState(seed)
    order = grid[rng.permutation(len(grid))]
    
    arr = np.empty((0, 3), dtype=np.int64)
    sq_norms_arr = np.empty(0, dtype=np.int64)
    
    for p in order:
        if time.time() > deadline:
            break
        sq_norm_p = (p ** 2).sum()
        if _check_add(arr, p, sq_norms_arr, sq_norm_p):
            if len(arr) == 0:
                arr = p[None, :]
                sq_norms_arr = np.array([sq_norm_p], dtype=np.int64)
            else:
                arr = np.vstack([arr, p])
                sq_norms_arr = np.append(sq_norms_arr, sq_norm_p)
    return arr

def _local_search(points, grid, deadline):
    """
    Perform local search to improve the set of points.
    1. Greedy addition of any fitting point.
    2. 1-for-1 swaps.
    3. 1-for-2 swaps.
    """
    if points is None or len(points) == 0:
        return points
    
    improved = points.copy()
    point_set = set(map(tuple, improved))
    sq_norms_improved = (improved ** 2).sum(axis=1)
    n_grid = len(grid)
    rng = np.random.RandomState(42)
    
    # Phase 1: Greedy addition
    added = True
    while added and time.time() < deadline - 2.0:
        added = False
        # Shuffled order for randomness
        grid_shuffled = grid[rng.permutation(n_grid)]
        for p in grid_shuffled:
            if time.time() > deadline - 2.0:
                break
            if tuple(p) in point_set:
                continue
            sq_norm_p = (p ** 2).sum()
            if _check_add(improved, p, sq_norms_improved, sq_norm_p):
                improved = np.vstack([improved, p])
                sq_norms_improved = np.append(sq_norms_improved, sq_norm_p)
                point_set.add(tuple(p))
                added = True
    
    # Phase 2: Local Search Swaps
    for _iter in range(20):
        if time.time() > deadline - 2.0:
            break
        n = len(improved)
        if n == 0:
            break
        
        # 1-for-1 Swaps
        swap_improved = False
        max_attempts = 1000
        for attempts in range(max_attempts):
            if time.time() > deadline - 2.0:
                break
            i = rng.randint(n)
            p_remove = improved[i]
            p_add = grid[rng.randint(n_grid)]
            if tuple(p_add) in point_set:
                continue
            new_improved = np.delete(improved, i, axis=0)
            new_sq_norms = np.delete(sq_norms_improved, i)
            sq_norm_add = (p_add ** 2).sum()
            if _check_add(new_improved, p_add, new_sq_norms, sq_norm_add):
                improved = np.vstack([new_improved, p_add])
                sq_norms_improved = np.append(new_sq_norms, sq_norm_add)
                point_set.remove(tuple(p_remove))
                point_set.add(tuple(p_add))
                swap_improved = True
                n = len(improved)
                break
        
        # 1-for-2 Swaps
        if not swap_improved and n > 0:
            max_attempts = 500
            for attempts in range(max_attempts):
                if time.time() > deadline - 2.0:
                    break
                i = rng.randint(n)
                p_remove = improved[i]
                new_improved = np.delete(improved, i, axis=0)
                new_sq_norms = np.delete(sq_norms_improved, i)
                
                # Try to add 2 points
                for _ in range(10):
                    if time.time() > deadline - 2.0:
                        break
                    idx1 = rng.randint(n_grid)
                    idx2 = rng.randint(n_grid)
                    if idx1 == idx2:
                        continue
                    p_add1 = grid[idx1]
                    p_add2 = grid[idx2]
                    if tuple(p_add1) in point_set or tuple(p_add2) in point_set:
                        continue
                    
                    sq_norm_add1 = (p_add1 ** 2).sum()
                    if not _check_add(new_improved, p_add1, new_sq_norms, sq_norm_add1):
                        continue
                    
                    temp_improved = np.vstack([new_improved, p_add1])
                    temp_sq_norms = np.append(new_sq_norms, sq_norm_add1)
                    
                    sq_norm_add2 = (p_add2 ** 2).sum()
                    if _check_add(temp_improved, p_add2, temp_sq_norms, sq_norm_add2):
                        improved = np.vstack([temp_improved, p_add2])
                        sq_norms_improved = np.append(temp_sq_norms, sq_norm_add2)
                        point_set.remove(tuple(p_remove))
                        point_set.add(tuple(p_add1))
                        point_set.add(tuple(p_add2))
                        swap_improved = True
                        n = len(improved)
                        break
                if swap_improved:
                    break
    return improved

def no_5_on_a_sphere8() -> np.ndarray:
    """
    Returns a numpy array of points from the 8x8x8 grid such that no 5 are co-spherical.
    """
    N = 8
    grid = np.array(list(itertools.product(range(N), range(N), range(N))), dtype=np.int64)
    sq_norms_all = (grid ** 2).sum(axis=1)
    
    start_time = time.time()
    deadline = start_time + 600.0  # Safety margin
    
    best_set = None
    best_len = 0
    candidates = []
    
    # Phase 1: Parallel-like greedy passes (single threaded but many)
    # We will run as many greedy passes as we can in ~300s
    greedy_deadline = start_time + 300.0
    seed = 0
    while time.time() < greedy_deadline:
        result = _greedy_pass(seed, grid, sq_norms_all, greedy_deadline)
        seed += 1
        if result is not None and len(result) > 0:
            candidates.append(result)
            candidates.sort(key=lambda x: len(x), reverse=True)
            if len(candidates) > 10:
                candidates.pop()
            if len(result) > best_len:
                best_len = len(result)
                best_set = result.copy()
    
    # Phase 2: Local search on top candidates
    if candidates:
        # Allocate remaining time among top 3 candidates
        top_candidates = candidates[:3]
        time_per_candidate = (deadline - time.time() - 5.0) / len(top_candidates)
        for i, cand in enumerate(top_candidates):
            t_start = time.time()
            t_deadline = t_start + time_per_candidate
            improved = _local_search(cand, grid, t_deadline)
            if len(improved) > len(best_set):
                best_set = improved
                best_len = len(improved)
    
    if best_set is None:
        return np.array([[0, 0, 0]], dtype=np.int64)
    
    return best_set.astype(np.int64)

# EVOLVE-BLOCK-END
