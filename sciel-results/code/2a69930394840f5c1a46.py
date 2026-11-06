import time
from itertools import combinations

def _subset_sums_distinct(arr):
    """Check if all 2^n subset sums are distinct."""
    n = len(arr)
    sums = [0] * (1 << n)
    # Use bit manipulation to compute all subset sums
    for i in range(1, 1 << n):
        # Find lowest set bit
        lsb = i & (-i)
        bit_idx = lsb.bit_length() - 1
        sums[i] = sums[i ^ lsb] + arr[bit_idx]
    return len(set(sums)) == (1 << n)

def _try_reduce(arr, max_time):
    """
    Try to reduce the largest element by local search.
    Start from powers of 2 and try to decrease elements while maintaining
    distinct subset sums property.
    """
    start_time = time.time()
    n = len(arr)
    best = list(arr)
    best_max = max(best)
    
    # Sort the array
    best.sort()
    
    # Try to reduce each element, starting from the largest
    # We'll do a greedy approach: try to reduce each element as much as possible
    # while keeping the property
    
    improved = True
    while improved:
        improved = False
        for i in range(n - 1, -1, -1):  # From largest to smallest
            if time.time() - start_time > max_time:
                break
            # Try to reduce element i
            # It must stay >= best[i-1] + 1 if i > 0, and >= 1
            min_val = 1 if i == 0 else best[i - 1] + 1
            # Try values from current down to min_val
            for val in range(best[i] - 1, min_val - 1, -1):
                candidate = list(best)
                candidate[i] = val
                if _subset_sums_distinct(candidate):
                    best = candidate
                    best_max = max(best)
                    improved = True
                    break
    
    return best

def _try_smart_search(arr, max_time):
    """
    More sophisticated search: try to find a better solution by
    perturbing the powers of 2 solution.
    """
    start_time = time.time()
    n = len(arr)
    best = list(arr)
    best.sort()
    best_max = max(best)
    
    # Try random perturbations
    import random
    random.seed(42)
    
    for _ in range(10000):
        if time.time() - start_time > max_time:
            break
        
        candidate = list(best)
        # Pick a random index to modify
        idx = random.randint(0, n - 1)
        # Try a random new value
        min_val = 1 if idx == 0 else candidate[idx - 1] + 1
        if idx < n - 1:
            max_val = candidate[idx + 1] - 1
        else:
            max_val = 1048576
        
        if min_val > max_val:
            continue
            
        new_val = random.randint(min_val, max_val)
        if new_val == candidate[idx]:
            continue
        
        candidate[idx] = new_val
        
        # Sort to maintain order
        candidate.sort()
        
        if _subset_sums_distinct(candidate):
            if max(candidate) < best_max:
                best = candidate
                best_max = max(best)
    
    return best

def construct():
    """
    Returns a set of 12 distinct positive integers with distinct subset sums.
    Uses local search to try to find a solution with a smaller largest element
    than the powers of 2 solution.
    """
    start_time = time.time()
    deadline = 80  # Safety margin before 90s limit
    
    # Start with powers of 2
    base = [1 << i for i in range(12)]
    
    # First, try greedy reduction
    result = _try_reduce(base, deadline - time.time() - 10)
    
    # If we have time, try smart search
    if time.time() - start_time < deadline:
        result2 = _try_smart_search(result, deadline - time.time() - 5)
        if max(result2) < max(result):
            result = result2
    
    return result
