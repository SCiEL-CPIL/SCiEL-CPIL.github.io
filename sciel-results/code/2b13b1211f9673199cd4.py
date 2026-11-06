"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _all_subset_sums_distinct(arr):
    """Check if all 2^n subset sums are distinct."""
    n = len(arr)
    sums = set()
    # Use bit representation: for each mask 0..(2^n - 1), compute sum
    for mask in range(1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += arr[i]
        if s in sums:
            return False
        sums.add(s)
    return True

def _local_search(initial, deadline):
    """Local search to minimize the largest element while keeping distinct subset sums."""
    best = list(initial)
    best_max = max(best)
    start_time = time.time()
    
    # Sort the initial array
    arr = sorted(best)
    
    # Try to reduce each element, starting from the largest
    # We'll do iterative local search
    improved = True
    while improved and (time.time() - start_time) < deadline:
        improved = False
        # Try reducing each element
        for i in range(len(arr) - 1, -1, -1):
            if time.time() - start_time > deadline:
                break
            # Try decreasing arr[i] by 1, 2, etc.
            for delta in range(1, min(arr[i] - 1, 50) + 1):
                if time.time() - start_time > deadline:
                    break
                new_arr = list(arr)
                new_val = arr[i] - delta
                if new_val < 1:
                    break
                # Check if new_val is already in the set (distinctness of elements)
                if new_val in new_arr:
                    continue
                new_arr[i] = new_val
                # Keep sorted
                new_arr.sort()
                # Check all subset sums distinct
                if _all_subset_sums_distinct(new_arr):
                    arr = new_arr
                    best = list(arr)
                    best_max = max(arr)
                    improved = True
                    break
    
    return best

def construct():
    # Start with powers of 2: max = 2048
    initial = [1 << i for i in range(12)]
    
    start_time = time.time()
    deadline = start_time + 80  # 80 seconds to be safe
    
    # Local search to reduce the maximum
    result = _local_search(initial, deadline)
    
    return sorted(result)
# EVOLVE-BLOCK-END
