import time
import sys
import random

sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    # Pool of candidates
    # Larger numbers are safer.
    pool = list(range(256, 128, -1))
    
    def check_addition(current_sums, x):
        """
        Returns new_sums set if valid, else None.
        """
        # Generate new sums
        # Optimization: list comprehension is fast
        new_sums_list = [s + x for s in current_sums]
        new_sums_list.append(x)
        
        # Check distinctness
        old_set = current_sums
        new_set = set(new_sums_list)
        if old_set.intersection(new_set):
            return None
            
        # Check divisibility
        # 1. New vs New
        # Sort to ensure we only check larger % smaller
        new_sums_sorted = sorted(new_sums_list)
        n_new = len(new_sums_sorted)
        for i in range(n_new):
            s_i = new_sums_sorted[i]
            # If s_i is 0, skip (not possible)
            for j in range(i + 1, n_new):
                s_j = new_sums_sorted[j]
                if s_j % s_i == 0:
                    return None
                    
        # 2. New vs Old
        # For each n in new_set:
        #   Check against all s in old_set
        for n in new_set:
            for s in old_set:
                if n > s:
                    if n % s == 0:
                        return None
                else:
                    if s % n == 0:
                        return None
                        
        return old_set.union(new_set)

    def search(current_elements, current_sums, candidates, target_size):
        if time.time() > deadline:
            return None
        
        if len(current_elements) == target_size:
            return list(current_elements)
        
        if len(candidates) < target_size - len(current_elements):
            return None
        
        # Heuristic: Try candidates in given order
        for i, x in enumerate(candidates):
            new_sums = check_addition(current_sums, x)
            if new_sums is not None:
                remaining = candidates[i+1:]
                res = search(current_elements + [x], new_sums, remaining, target_size)
                if res is not None:
                    return res
                    
        return None

    best_set = []
    
    # Try sizes 10 down to 1
    for k in range(10, 0, -1):
        if time.time() > deadline:
            break
            
        found = False
        attempts = 0
        max_attempts = 10 # Reduce attempts for larger k to save time
        
        while attempts < max_attempts and time.time() < deadline:
            pool_copy = list(pool)
            if attempts > 0:
                random.shuffle(pool_copy)
            
            res = search([], set(), pool_copy, k)
            if res is not None:
                best_set = res
                found = True
                break
            attempts += 1
            
        if found:
            break
            
    if not best_set:
        best_set = [256]
        
    return best_set
