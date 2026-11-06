"""
Rewritten program: search for a 12-element set of distinct positive integers
with all 4096 subset sums distinct, minimizing the largest element.

The seed solution [1,2,4,...,2048] has max 2048. We try to find a solution
with max < 2048 via randomized local search, with a strict time guard.
"""
import time
import random

def construct():
    deadline = time.time() + 80.0  # 80s budget, leaving 10s safety margin
    n = 12
    
    # Validate: all 4096 subset sums must be distinct
    def subset_sums_distinct(arr):
        n_local = len(arr)
        sums = [0] * (1 << n_local)
        for i in range(n_local):
            v = arr[i]
            bit = 1 << i
            for mask in range(1 << i):
                sums[mask | bit] = sums[mask] + v
        return len(set(sums)) == (1 << n_local)
    
    def subset_sums_distinct_fast(arr):
        """Faster check using iterative construction, returns bool."""
        n_local = len(arr)
        total = 1 << n_local
        # Use a set; for n=12, 4096 elements is small
        sums = {0}
        for v in arr:
            sums = sums | {s + v for s in sums}
            if len(sums) < len(sums) + 1:  # always true, but structure helps
                pass
            if len(sums) != (1 << (arr.index(v) + 1)):
                return False
        return len(sums) == total
    
    def is_valid(arr):
        if len(set(arr)) != len(arr):
            return False
        if any(x <= 0 for x in arr):
            return False
        return subset_sums_distinct_fast(arr)
    
    # Start with the known solution
    best = [1 << i for i in range(n)]
    best_max = best[-1]
    
    def try_improve(start_arr, time_limit):
        """Randomized local search: try to reduce max element."""
        current = list(start_arr)
        current_max = max(current)
        local_best = current
        local_best_max = current_max
        t_start = time.time()
        
        while time.time() - t_start < time_limit:
            if time.time() > time_limit:
                break
            
            # Try to reduce the largest element
            # Pick a random position to try to decrease
            idx = random.randint(0, n - 1)
            old_val = current[idx]
            
            # Try values from old_val - 1 down to 1
            improved = False
            for new_val in range(old_val - 1, 0, -1):
                if new_val in current:
                    continue
                current[idx] = new_val
                if is_valid(current):
                    improved = True
                    break
            
            if improved:
                if max(current) < local_best_max:
                    local_best = list(current)
                    local_best_max = max(current)
            else:
                current[idx] = old_val
            
            # Occasionally try a random perturbation to escape local optima
            if random.random() < 0.1:
                idx2 = random.randint(0, n - 1)
                old_val2 = current[idx2]
                # Try a random new value
                for _ in range(20):
                    new_val2 = random.randint(1, local_best_max)
                    if new_val2 in current:
                        continue
                    current[idx2] = new_val2
                    if is_valid(current):
                        break
                    current[idx2] = old_val2
                else:
                    current[idx2] = old_val2
        
        return local_best, local_best_max
    
    # Try several different starting points and search strategies
    # Strategy 1: Start from powers of 2, greedily try to reduce each element
    arr = list(best)
    # Sort and try to reduce from the largest
    arr.sort()
    for i in range(n - 1, -1, -1):
        if time.time() > deadline:
            break
        for v in range(arr[i] - 1, 0, -1):
            if v in arr:
                continue
            arr[i] = v
            if is_valid(arr):
                break
            arr[i] = best[i]
    
    if is_valid(arr) and max(arr) < best_max:
        best = list(arr)
        best_max = max(arr)
    
    # Strategy 2: Randomized restarts from perturbed powers of 2
    for restart in range(50):
        if time.time() > deadline:
            break
        # Create a perturbed starting point
        start = list(best)
        for i in range(n):
            if random.random() < 0.3:
                start[i] = max(1, start[i] - random.randint(1, max(1, start[i] // 2)))
        start.sort()
        
        # Try to improve from this starting point
        local_best, local_max = try_improve(start, 1.0)
        if local_max < best_max:
            best = local_best
            best_max = local_max
    
    # Final check
    if is_valid(best):
        return best
    
    # Fallback to known solution
    return [1 << i for i in range(n)]
