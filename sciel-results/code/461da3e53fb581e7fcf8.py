# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time
import concurrent.futures
import os

# Global variables for worker processes
_GRID = None
_N = 8
_SQ_NORMS = None  # Pre-computed squared norms for grid points

def _worker_init(grid_data, sq_norms_data):
    """
    Initialize global variables in the worker process.
    """
    global _GRID, _SQ_NORMS
    _GRID = grid_data
    _SQ_NORMS = sq_norms_data

def _check_can_add(points, p, sq_norms_points, sq_norm_p):
    """
    Check if point p can be added to the set of points.
    points: (m, 3) array
    p: (3,) array
    sq_norms_points: (m,) array of squared norms of points
    sq_norm_p: scalar squared norm of p
    """
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

def _run_greedy(seed):
    """
    Run a single greedy pass with a specific random seed.
    Returns the best set of points found.
    """
    global _GRID, _SQ_NORMS
    grid = _GRID
    sq_norms_all = _SQ_NORMS
    
    rng = np.random.RandomState(seed)
    order = grid[rng.permutation(len(grid))]
    
    arr = np.empty((0, 3), dtype=np.int64)
    sq_norms_arr = np.empty(0, dtype=np.int64)
    
    for p in order:
        sq_norm_p = (p ** 2).sum()
        
        if _check_can_add(arr, p, sq_norms_arr, sq_norm_p):
            if len(arr) == 0:
                arr = p[None, :]
                sq_norms_arr = np.array([sq_norm_p], dtype=np.int64)
            else:
                arr = np.vstack([arr, p])
                sq_norms_arr = np.append(sq_norms_arr, sq_norm_p)
                
    return arr

def _local_search(points, deadline):
    """
    Perform local search to improve the set of points.
    Strategy:
    1. Try to add any remaining point that fits.
    2. 1-for-1 swaps.
    3. 1-for-2 swaps (remove one, try to add two).
    """
    if points is None or len(points) == 0:
        return points
        
    improved = points.copy()
    point_set = set(map(tuple, improved))
    sq_norms_improved = (improved ** 2).sum(axis=1)
    
    # Phase 1: Greedy addition of any fitting point
    added = True
    while added and time.time() < deadline - 5.0:
        added = False
        for p in _GRID:
            if tuple(p) in point_set:
                continue
            sq_norm_p = (p ** 2).sum()
            if _check_can_add(improved, p, sq_norms_improved, sq_norm_p):
                improved = np.vstack([improved, p])
                sq_norms_improved = np.append(sq_norms_improved, sq_norm_p)
                point_set.add(tuple(p))
                added = True
    
    # Phase 2: Local Search Swaps
    rng = np.random.RandomState(123)
    n_grid = len(_GRID)
    
    # We will iterate a few times
    for _iter in range(10):
        if time.time() > deadline - 5.0:
            break
        
        n = len(improved)
        if n == 0:
            break
            
        # 1-for-1 Swaps
        swap_improved = False
        # Try all removals? No, too slow. Random sample.
        # But n is small (~10-20). We can try all removals for each candidate?
        # Number of candidates is ~500. 20 * 500 = 10,000 checks.
        # Each check is O(m^4) determinants? No, _check_can_add is O(m^4) but m is small.
        # Actually _check_can_add checks all 4-combinations of existing points.
        # If m=10, C(10,4)=210. 10000 * 210 dets = 2.1M dets. Might be slow in Python.
        # Let's stick to random swaps but increase count.
        
        attempts = 0
        max_attempts = 500
        while attempts < max_attempts and time.time() < deadline - 5.0:
            attempts += 1
            i = rng.randint(n)
            p_remove = improved[i]
            p_add = _GRID[rng.randint(n_grid)]
            
            if tuple(p_add) in point_set:
                continue
                
            new_improved = np.delete(improved, i, axis=0)
            new_sq_norms = np.delete(sq_norms_improved, i)
            
            sq_norm_add = (p_add ** 2).sum()
            if _check_can_add(new_improved, p_add, new_sq_norms, sq_norm_add):
                # Swap succeeded
                improved = np.vstack([new_improved, p_add])
                sq_norms_improved = np.append(new_sq_norms, sq_norm_add)
                point_set.remove(tuple(p_remove))
                point_set.add(tuple(p_add))
                swap_improved = True
                n = len(improved)
                break # Restart random sampling after a change? Or just continue?
                # Restarting helps escape local minima.
        
        # 1-for-2 Swaps (Remove 1, Add 2)
        # This is harder. Remove i. Try to add two points a, b.
        # We can try to greedily add the best 2 points after removal.
        if not swap_improved and n > 0:
            attempts = 0
            max_attempts = 200
            while attempts < max_attempts and time.time() < deadline - 5.0:
                attempts += 1
                i = rng.randint(n)
                p_remove = improved[i]
                
                new_improved = np.delete(improved, i, axis=0)
                new_sq_norms = np.delete(sq_norms_improved, i)
                
                # Try to add 2 points from the remaining grid
                # We can pick 2 random points and see if they fit.
                # Or, greedily pick the best 2.
                # Let's try a few random pairs.
                for _ in range(5):
                    idx1 = rng.randint(n_grid)
                    idx2 = rng.randint(n_grid)
                    if idx1 == idx2: continue
                    p_add1 = _GRID[idx1]
                    p_add2 = _GRID[idx2]
                    
                    if tuple(p_add1) in point_set or tuple(p_add2) in point_set:
                        continue
                        
                    # Check if p_add1 fits
                    sq_norm_add1 = (p_add1 ** 2).sum()
                    if not _check_can_add(new_improved, p_add1, new_sq_norms, sq_norm_add1):
                        continue
                        
                    # Tentatively add p_add1
                    temp_improved = np.vstack([new_improved, p_add1])
                    temp_sq_norms = np.append(new_sq_norms, sq_norm_add1)
                    
                    # Check if p_add2 fits with p_add1 added
                    sq_norm_add2 = (p_add2 ** 2).sum()
                    if _check_can_add(temp_improved, p_add2, temp_sq_norms, sq_norm_add2):
                        # Success!
                        improved = temp_improved
                        improved = np.vstack([improved, p_add2])
                        sq_norms_improved = temp_sq_norms
                        sq_norms_improved = np.append(sq_norms_improved, sq_norm_add2)
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
    global _GRID, _N, _SQ_NORMS
    _N = 8
    _GRID = np.array(list(itertools.product(range(_N), range(_N), range(_N))), dtype=np.int64)
    _SQ_NORMS = (_GRID ** 2).sum(axis=1)
    
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin
    
    # Determine number of CPUs
    n_cpus = os.cpu_count() or 4
    n_workers = min(n_cpus, 8)  # Cap at 8 to avoid overhead
    
    best_set = None
    best_len = 0
    
    # Keep a list of top candidates for local search
    candidates = []
    
    # Phase 1: Parallel greedy passes
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=n_workers, 
        initializer=_worker_init, 
        initargs=(_GRID, _SQ_NORMS)
    ) as executor:
        
        # We'll generate an infinite stream of seeds
        seed_gen = iter(range(100000))
        
        # Keep track of futures
        running_futures = []
        
        while time.time() < deadline - 120.0:  # Leave 120s for local search
            # Submit new tasks if we have capacity
            while len(running_futures) < n_workers * 2 and time.time() < deadline - 120.0:
                try:
                    seed = next(seed_gen)
                except StopIteration:
                    break
                future = executor.submit(_run_greedy, seed)
                running_futures.append(future)
            
            if not running_futures:
                break
            
            # Check for completed futures
            done, not_done = concurrent.futures.wait(running_futures, timeout=1.0)
            
            for f in done:
                running_futures.remove(f)
                try:
                    result = f.result()
                    if result is not None and len(result) > 0:
                        # Add to candidates if it's distinct and good
                        # We just keep the top few by length
                        candidates.append(result)
                        # Sort and keep top 5
                        candidates.sort(key=lambda x: len(x), reverse=True)
                        if len(candidates) > 5:
                            candidates.pop()
                            
                        if len(result) > best_len:
                            best_len = len(result)
                            best_set = result.copy()
                except Exception as e:
                    pass
            
            if not running_futures:
                 time.sleep(0.1)
    
    # Phase 2: Local search on the best candidates
    # We have ~120s left.
    # Allocate time per candidate.
    if candidates:
        time_per_candidate = (deadline - time.time() - 10.0) / len(candidates)
        for i, cand in enumerate(candidates):
            # Skip if same as best_set already processed?
            # Just process top 3 distinct ones.
            # Simple check: if we already processed a set with same points, skip.
            # For simplicity, just process the top 3 unique by length.
            # The list is sorted, so top ones are first.
            if i >= 3:
                break
                
            # Avoid processing duplicates
            if i > 0:
                prev = candidates[i-1]
                # Simple check: if same length and same sum of coords?
                # Just process it, local search is idempotent-ish.
                pass
            
            t_start = time.time()
            t_deadline = t_start + time_per_candidate
            improved = _local_search(cand, t_deadline)
            
            if len(improved) > len(best_set):
                best_set = improved
                best_len = len(improved)
    
    if best_set is None:
        return np.array([[0, 0, 0]], dtype=np.int64)
        
    return best_set.astype(np.int64)

# EVOLVE-BLOCK-END
