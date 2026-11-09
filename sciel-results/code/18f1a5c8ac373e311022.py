"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools

# EVOLVE-BLOCK-START

def _check_distinct_subset_sums(arr):
    """Check if all 2^n subset sums are distinct."""
    n = len(arr)
    sums = set()
    # Use bitmask enumeration for n=12 (4096 subsets)
    for mask in range(1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += arr[i]
        if s in sums:
            return False
        sums.add(s)
    return True

def _check_distinct_subset_sums_fast(arr):
    """Faster check using numpy for subset sums."""
    import numpy as np
    n = len(arr)
    arr = np.array(arr)
    # Generate all subset sums
    # For n=12, we can do this efficiently
    sums = [0]
    for a in arr:
        sums = sums + [s + a for s in sums]
    return len(sums) == len(set(sums))

def _greedy_construct(n=12):
    """
    Greedy construction: at each step, pick the smallest positive integer
    that keeps all subset sums distinct.
    """
    arr = []
    for i in range(n):
        # Try candidates starting from 1
        candidate = 1
        while True:
            trial = arr + [candidate]
            if _check_distinct_subset_sums_fast(trial):
                arr.append(candidate)
                break
            candidate += 1
    return arr

def _local_search(arr, max_iters=5000, timeout=10):
    """
    Local search: try to reduce the largest element by swapping or adjusting.
    """
    start_time = time.time()
    n = len(arr)
    best = arr[:]
    best_max = max(best)
    
    for iteration in range(max_iters):
        if time.time() - start_time > timeout:
            break
        
        # Try to reduce the largest element
        arr = best[:]
        max_idx = arr.index(max(arr))
        
        # Try replacing the largest with something smaller
        improved = False
        for new_val in range(1, arr[max_idx]):
            trial = arr[:]
            trial[max_idx] = new_val
            if len(set(trial)) != n:
                continue
            if _check_distinct_subset_sums_fast(trial):
                best = trial
                best_max = max(best)
                improved = True
                break
        
        if not improved:
            # Try other local moves: swap two elements
            improved = False
            for i in range(n):
                for j in range(i+1, n):
                    trial = best[:]
                    trial[i], trial[j] = trial[j], trial[i]
                    if _check_distinct_subset_sums_fast(trial):
                        # Only accept if it helps reduce max
                        if max(trial) < best_max:
                            best = trial
                            best_max = max(best)
                            improved = True
                            break
                if improved:
                    break
            
            if not improved:
                # Try small perturbations
                if random.random() < 0.3:
                    idx = random.randint(0, n-1)
                    delta = random.choice([-2, -1, 1, 2])
                    new_val = best[idx] + delta
                    if new_val > 0:
                        trial = best[:]
                        trial[idx] = new_val
                        if len(set(trial)) == n and _check_distinct_subset_sums_fast(trial):
                            if max(trial) <= best_max:
                                best = trial
                                best_max = max(best)
                else:
                    break  # No improvement, stop
    
    return best

def _simulated_annealing(arr, timeout=15):
    """
    Simulated annealing to minimize the largest element.
    """
    import math
    start_time = time.time()
    n = len(arr)
    current = arr[:]
    current_max = max(current)
    best = current[:]
    best_max = current_max
    
    T = 100.0
    cooling = 0.999
    
    while T > 0.1 and time.time() - start_time < timeout:
        # Generate a neighbor
        neighbor = current[:]
        move_type = random.random()
        
        if move_type < 0.4:
            # Adjust one element
            idx = random.randint(0, n-1)
            delta = random.randint(-5, 5)
            new_val = neighbor[idx] + delta
            if new_val > 0:
                neighbor[idx] = new_val
        elif move_type < 0.7:
            # Swap two elements
            i, j = random.sample(range(n), 2)
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
        else:
            # Replace one element with a random smaller value
            idx = random.randint(0, n-1)
            if neighbor[idx] > 1:
                neighbor[idx] = random.randint(1, neighbor[idx])
        
        # Check validity
        if len(set(neighbor)) != n:
            T *= cooling
            continue
        
        if not _check_distinct_subset_sums_fast(neighbor):
            T *= cooling
            continue
        
        neighbor_max = max(neighbor)
        
        # Accept if better, or with probability based on temperature
        if neighbor_max < current_max:
            current = neighbor
            current_max = neighbor_max
            if current_max < best_max:
                best = current[:]
                best_max = current_max
        else:
            # Accept worse solutions with some probability
            delta = neighbor_max - current_max
            if delta > 0:
                prob = math.exp(-delta / max(T, 0.01))
                if random.random() < prob:
                    current = neighbor
                    current_max = neighbor_max
        
        T *= cooling
    
    return best

def construct():
    """
    Construct exactly 12 distinct positive integers <= 1048576 such that
    all 4096 subset sums are distinct. Minimize the largest integer.
    """
    start_time = time.time()
    deadline = 80  # Leave 10s margin for safety
    
    # Start with greedy construction
    arr = _greedy_construct(12)
    best = arr[:]
    best_max = max(best)
    
    # Verify
    assert _check_distinct_subset_sums_fast(best), "Greedy failed"
    
    # Try local search
    if time.time() - start_time < deadline - 15:
        ls_result = _local_search(best, max_iters=2000, timeout=10)
        if max(ls_result) < best_max:
            best = ls_result
            best_max = max(best)
    
    # Try simulated annealing from best
    if time.time() - start_time < deadline - 10:
        sa_result = _simulated_annealing(best, timeout=15)
        if max(sa_result) < best_max:
            best = sa_result
            best_max = max(best)
    
    # Final verification
    assert len(best) == 12, f"Expected 12 elements, got {len(best)}"
    assert all(x > 0 for x in best), "All elements must be positive"
    assert len(set(best)) == 12, "All elements must be distinct"
    assert max(best) <= 1048576, f"Max element {max(best)} exceeds bound"
    assert _check_distinct_subset_sums_fast(best), "Subset sums not distinct"
    
    return best

# EVOLVE-BLOCK-END
