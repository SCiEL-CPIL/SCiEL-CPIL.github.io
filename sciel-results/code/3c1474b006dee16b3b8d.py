# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time
import concurrent.futures
import os

# Global variables for worker processes
_GRID = None
_N = 8
_SQ_NORMS = None

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

def _simulated_annealing(initial_points, deadline):
    """
    Perform simulated annealing starting from initial_points.
    Allows removing points to escape local maxima.
    """
    if initial_points is None or len(initial_points) == 0:
        return initial_points
        
    points = initial_points.copy()
    point_set = set(map(tuple, points))
    sq_norms_points = (points ** 2).sum(axis=1)
    
    best_points = points.copy()
    best_len = len(points)
    
    # Temperature schedule
    T_start = 1.0
    T_end = 0.01
    alpha = 0.95
    
    T = T_start
    rng = np.random.RandomState(42)
    
    while T > T_end and time.time() < deadline - 5.0:
        T *= alpha
        
        # Generate a random move
        move_type = rng.randint(0, 3)  # 0: add, 1: remove, 2: swap
        
        if move_type == 0:
            # Try to add a random point
            candidates = [p for p in _GRID if tuple(p) not in point_set]
            if not candidates:
                continue
            p = candidates[rng.randint(0, len(candidates))]
            sq_norm_p = (p ** 2).sum()
            
            if _check_can_add(points, p, sq_norms_points, sq_norm_p):
                # Accept
                points = np.vstack([points, p])
                point_set.add(tuple(p))
                sq_norms_points = np.append(sq_norms_points, sq_norm_p)
                if len(points) > best_len:
                    best_len = len(points)
                    best_points = points.copy()
            else:
                # Reject (adding is never worse, so if it doesn't fit, skip)
                pass
                
        elif move_type == 1:
            # Try to remove a random point
            if len(points) <= 1:
                continue
            idx = rng.randint(0, len(points))
            p = points[idx]
            
            # Remove
            points = np.delete(points, idx, axis=0)
            point_set.remove(tuple(p))
            sq_norms_points = np.delete(sq_norms_points, idx)
            
            # With some probability, accept the removal (to escape local maxima)
            if rng.rand() < 0.5:
                # Keep removed
                pass
            else:
                # Re-add
                points = np.vstack([points, p])
                point_set.add(tuple(p))
                sq_norms_points = np.append(sq_norms_points, (p ** 2).sum())
                
        else:
            # Try to swap: remove one, add one
            if len(points) == 0:
                continue
            idx = rng.randint(0, len(points))
            p_old = points[idx]
            
            candidates = [p for p in _GRID if tuple(p) not in point_set]
            if not candidates:
                continue
            p_new = candidates[rng.randint(0, len(candidates))]
            
            # Try swap
            temp_points = np.delete(points, idx, axis=0)
            temp_sq_norms = np.delete(sq_norms_points, idx)
            sq_norm_new = (p_new ** 2).sum()
            
            if _check_can_add(temp_points, p_new, temp_sq_norms, sq_norm_new):
                # Accept swap
                points = np.vstack([temp_points, p_new])
                point_set.remove(tuple(p_old))
                point_set.add(tuple(p_new))
                sq_norms_points = np.append(temp_sq_norms, sq_norm_new)
                if len(points) > best_len:
                    best_len = len(points)
                    best_points = points.copy()
                    
    # Final pass: try to add any remaining point that fits
    added = True
    while added and time.time() < deadline - 2.0:
        added = False
        for p in _GRID:
            if tuple(p) in point_set:
                continue
            sq_norm_p = (p ** 2).sum()
            if _check_can_add(points, p, sq_norms_points, sq_norm_p):
                points = np.vstack([points, p])
                point_set.add(tuple(p))
                sq_norms_points = np.append(sq_norms_points, sq_norm_p)
                added = True
                if len(points) > best_len:
                    best_len = len(points)
                    best_points = points.copy()
                    
    return best_points

def no_5_on_a_sphere8() -> np.ndarray:
    """
    Returns a numpy array of points from the 8x8x8 grid such that no 5 are co-spherical.
    """
    global _GRID, _SQ_NORMS, _N
    _N = 8
    _GRID = np.array(list(itertools.product(range(_N), range(_N), range(_N))), dtype=np.int64)
    _SQ_NORMS = (_GRID ** 2).sum(axis=1)
    
    start_time = time.time()
    deadline = start_time + 550.0 # Safety margin
    
    # Determine number of CPUs
    n_cpus = os.cpu_count() or 4
    n_workers = min(n_cpus, 8) # Cap at 8 to avoid overhead
    
    best_set = None
    best_len = 0
    
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
        
        while time.time() < deadline - 100.0: # Leave 100s for local search/SA
            # Submit new tasks if we have capacity
            # Keep up to n_workers * 2 tasks in flight
            while len(running_futures) < n_workers * 2 and time.time() < deadline - 100.0:
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
                    if result is not None and len(result) > best_len:
                        best_len = len(result)
                        best_set = result.copy()
                except Exception as e:
                    # Handle worker errors
                    pass
            
            # If no tasks are running and no more to submit, break
            if not running_futures and time.time() >= deadline - 100.0:
                 break
                 
            # Small sleep to avoid busy waiting if we are not submitting
            if not running_futures:
                 time.sleep(0.1)
    
    # Phase 2: Improve the best set with Simulated Annealing in the main process
    if best_set is not None and time.time() < deadline - 10.0:
        improved = _simulated_annealing(best_set, deadline)
        if improved is not None and len(improved) > len(best_set):
            best_set = improved

    if best_set is None:
        # Fallback: return an empty set or a small valid set
        # A single point is always valid.
        return np.array([[0, 0, 0]], dtype=np.int64)
        
    return best_set.astype(np.int64)

# EVOLVE-BLOCK-END
