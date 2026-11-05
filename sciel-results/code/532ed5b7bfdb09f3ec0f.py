"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _check_distinct_subset_sums(arr):
    """Check if all subset sums are distinct. Returns True if valid."""
    n = len(arr)
    sums = set()
    sums.add(0)
    for x in arr:
        current_sums = set()
        for s in sums:
            new_s = s + x
            if new_s in sums or new_s in current_sums:
                return False
            current_sums.add(new_s)
        sums.update(current_sums)
    return True

def _try_local_search(arr, max_time):
    """Try to improve the array by reducing elements while maintaining distinct subset sums."""
    start_time = time.time()
    n = len(arr)
    best = list(arr)
    best_max = max(best)
    
    # Sort in ascending order for systematic search
    best = sorted(best)
    
    # Try to reduce the largest element first, then others
    for attempt in range(100000):
        if time.time() - start_time > max_time:
            break
        
        # Pick a random index to try reducing
        idx = random.randint(0, n - 1)
        current_val = best[idx]
        
        if current_val <= 1:
            continue
        
        # Try reducing by a random amount
        reduction = random.randint(1, min(current_val - 1, 1000))
        new_val = current_val - reduction
        
        if new_val < 1:
            continue
        
        # Ensure distinctness
        if new_val in best:
            continue
        
        # Create new array
        new_arr = list(best)
        new_arr[idx] = new_val
        
        # Check validity
        if _check_distinct_subset_sums(new_arr):
            best = new_arr
            best_max = max(best)
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Leave 5s margin
    
    # Start with binary solution
    arr = [1 << i for i in range(12)]
    best = list(arr)
    best_max = max(best)
    
    # Try local search to improve
    improved = _try_local_search(arr, 40)
    if max(improved) < best_max:
        best = improved
        best_max = max(best)
    
    # If time remains, try another round with different random seed
    if time.time() < deadline - 5:
        random.seed(int(time.time() * 1000) % (2**32))
        improved2 = _try_local_search(arr, 30)
        if max(improved2) < best_max:
            best = improved2
            best_max = max(improved2)
    
    # Return sorted for consistency
    return sorted(best)
# EVOLVE-BLOCK-END
