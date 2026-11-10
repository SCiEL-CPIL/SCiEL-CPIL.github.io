# EVOLVE-BLOCK-START
import numpy as np
import time
import concurrent.futures

def _calc_area(x):
    """
    Compute the area of the union of the 64 triangles for given positions x.
    n = 64
    """
    n = 64
    w = 1.0 / n
    
    # a_i = (i + 1) / 64
    a = (np.arange(n) + 1) / n
    
    # y_m = (m + 0.5) / 20000, m = 0..19999
    m = np.arange(20000)
    y = (m + 0.5) / 20000.0
    
    # L_i = x_i + a_i * y
    # Shape: (n, 20000)
    L = x[:, np.newaxis] + a[:, np.newaxis] * y[np.newaxis, :]
    
    # Sort L_i for each y_m
    L_sorted = np.sort(L, axis=0)
    
    # Calculate union length at each height
    diffs = np.diff(L_sorted, axis=0)
    added_lengths = np.minimum(diffs, w)
    
    # Sum over triangles (axis 0) and add the base width w
    union_lengths = w + np.sum(added_lengths, axis=0)
    
    # Area is the mean of union lengths
    area = np.mean(union_lengths)
    return area

def _optimize_one(x0, deadline):
    """
    Perform coordinate descent from x0 until deadline.
    """
    n = 64
    best_x = x0.copy()
    best_area = _calc_area(best_x)
    
    # Coordinate descent with adaptive step size
    step_size = 0.05
    no_improve_count = 0
    
    while time.time() < deadline:
        improved = False
        for i in range(n):
            if time.time() >= deadline:
                break
            
            # Try moving in + direction
            x_trial = best_x.copy()
            x_trial[i] += step_size
            area_trial = _calc_area(x_trial)
            
            if area_trial < best_area:
                best_x = x_trial
                best_area = area_trial
                improved = True
                no_improve_count = 0
            else:
                # Try moving in - direction
                x_trial = best_x.copy()
                x_trial[i] -= step_size
                area_trial = _calc_area(x_trial)
                
                if area_trial < best_area:
                    best_x = x_trial
                    best_area = area_trial
                    improved = True
                    no_improve_count = 0
            
        if not improved:
            no_improve_count += 1
            # Reduce step size if stuck
            step_size *= 0.8
            if step_size < 1e-5:
                break
        else:
            # If improved, keep or slightly increase step size
            step_size = min(step_size * 1.05, 0.1)
            
    return best_x, best_area

def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Optimize the base positions x for the Kakeya needle problem (n=64)
    using parallel multi-start coordinate descent.
    """
    n = 64
    start_time = time.time()
    timeout = 500.0  # Leave margin for 600s limit
    deadline = start_time + timeout
    
    # 1. Generate diverse initial guesses
    # Include the Keich construction and random starts
    j = np.arange(1, int(np.log2(n)) + 1)
    keich_positions = []
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (int(np.log2(n)) - len(eps), 0), "constant")
        keich_positions.append(np.sum((1 - j) * eps * (2.0 ** -j) / int(np.log2(n))))
    x_keich = np.array(keich_positions, dtype=np.float64)
    
    rng = np.random.default_rng(42)
    initial_guesses = [x_keich]
    for _ in range(15):
        # Random starts in a reasonable range
        x_rand = rng.uniform(-0.5, 1.5, n)
        initial_guesses.append(x_rand)
    
    # 2. Parallel optimization
    best_x = x_keich.copy()
    best_area = _calc_area(best_x)
    
    # Use a thread pool. Note: GIL may limit parallelism for pure Python, 
    # but NumPy operations release the GIL, so this should be effective.
    # We split the remaining time among the workers, or just let them run until the global deadline.
    
    # Since _optimize_one checks the global deadline, we can launch them all.
    # However, launching too many threads might cause overhead. 
    # Let's use a moderate number of workers, e.g., 8 or 16.
    # The problem size is small, so 16 workers is likely fine.
    
    n_workers = 16
    # We have 16 initial guesses. Let's optimize each one.
    # If we have more time, we could generate more random starts, but 16 is a good start.
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = []
        for x0 in initial_guesses:
            futures.append(executor.submit(_optimize_one, x0, deadline))
        
        for future in concurrent.futures.as_completed(futures):
            try:
                x_opt, area_opt = future.result()
                if area_opt < best_area:
                    best_x = x_opt
                    best_area = area_opt
            except Exception:
                pass
    
    # 3. Final refinement with the best result found
    # Run one more coordinate descent from the best result with a tighter deadline
    if time.time() < deadline - 10:
        x_refined, area_refined = _optimize_one(best_x, deadline)
        if area_refined < best_area:
            best_x = x_refined
            best_area = area_refined
            
    return best_x

# EVOLVE-BLOCK-END
