import time
import sys

sys.setrecursionlimit(10000)

def construct():
    """
    Find a maximum-size Sidon set (all pairwise sums a+b for a<=b distinct)
    containing 1,2,5,11 within [1,284].
    
    Uses Backtracking with Forward Checking and Density Pruning.
    """
    prescribed = [1, 2, 5, 11]
    used_sums = set()
    for i, x in enumerate(prescribed):
        for y in prescribed[i:]:
            used_sums.add(x + y)
    
    current = list(prescribed)
    best_set = list(current)
    best_size = len(current)
    start_time = time.time()
    time_limit = 115.0  # Leave 5s margin

    def is_valid(x, arr, used):
        for y in arr:
            if x + y in used:
                return False
        if 2 * x in used:
            return False
        return True

    def get_new_sums(x, arr):
        return {x + y for y in arr} | {2 * x}

    def max_additional_elements(current_max):
        if current_max >= 284:
            return 0
        return int((284 - current_max) ** 0.5) + 3

    def backtrack(arr, used, start_val):
        nonlocal best_set, best_size, start_time, time_limit
        
        if time.time() - start_time > time_limit:
            return

        if len(arr) > best_size:
            best_size = len(arr)
            best_set = list(arr)
            
        current_max = arr[-1]
        max_add = max_additional_elements(current_max)
        if len(arr) + max_add <= best_size:
            return

        for x in range(start_val, 285):
            if is_valid(x, arr, used):
                new_used = used | get_new_sums(x, arr)
                backtrack(arr + [x], new_used, x + 1)
                
            if time.time() - start_time > time_limit:
                return

    backtrack(current, used_sums, 12)
    
    return best_set
