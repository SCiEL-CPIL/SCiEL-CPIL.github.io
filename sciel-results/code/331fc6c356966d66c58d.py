"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill
    
    n = 12
    num_subsets = 1 << n
    
    def subset_sums(arr):
        """Compute all subset sums efficiently."""
        sums = [0]
        for x in arr:
            sums = sums + [s + x for s in sums]
        return sums
    
    def is_valid(arr):
        """Check if all subset sums are distinct and all elements are distinct positive integers."""
        if len(set(arr)) != len(arr):
            return False
        if any(x <= 0 for x in arr):
            return False
        sums = subset_sums(arr)
        return len(set(sums)) == num_subsets
    
    # Start with powers of 2
    best = [1 << i for i in range(n)]
    best_max = max(best)
    
    # Try to reduce elements greedily from largest to smallest
    def try_reduce(arr, deadline):
        """Try to reduce elements to minimize the max."""
        arr = sorted(arr, reverse=True)
        # Try to reduce each element
        for i in range(n):
            if time.time() > deadline - 5:
                break
            current = arr[i]
            # Binary search for the minimum valid value
            lo, hi = 1, current - 1
            best_val = current
            while lo <= hi and time.time() < deadline - 5:
                mid = (lo + hi) // 2
                test_arr = arr.copy()
                test_arr[i] = mid
                if is_valid(test_arr):
                    best_val = mid
                    hi = mid - 1
                else:
                    lo = mid + 1
            arr[i] = best_val
        return arr
    
    # Initial greedy reduction
    arr = try_reduce(best, deadline)
    if max(arr) < best_max:
        best = arr
        best_max = max(best)
    
    # Local search: try random perturbations
    while time.time() < deadline - 10:
        # Pick a random element and try to reduce it
        i = random.randint(0, n - 1)
        current = arr[i]
        if current <= 1:
            continue
        # Try reducing by a random amount
        reduction = random.randint(1, max(1, current // 4))
        new_val = current - reduction
        if new_val < 1:
            continue
        test_arr = arr.copy()
        test_arr[i] = new_val
        if is_valid(test_arr):
            arr[i] = new_val
            if max(arr) < best_max:
                best = arr.copy()
                best_max = max(best)
        else:
            # Try small random changes
            delta = random.randint(-5, 5)
            if delta == 0:
                continue
            new_val = current + delta
            if new_val < 1 or new_val > best_max:
                continue
            test_arr = arr.copy()
            test_arr[i] = new_val
            if is_valid(test_arr):
                arr[i] = new_val
                if max(arr) < best_max:
                    best = arr.copy()
                    best_max = max(best)
    
    # Final greedy pass
    if time.time() < deadline - 5:
        arr = try_reduce(best, deadline)
        if max(arr) < best_max:
            best = arr
            best_max = max(best)
    
    return sorted(best)

# EVOLVE-BLOCK-END
