import time
import random
import sys

sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80s limit to be safe (90s wall clock)
    
    # Helper to check if adding x to current_sums is valid
    def check_addition(current_sums, x):
        # Generate all new subset sums when adding x
        # New sums are: x, and for each s in current_sums, s+x
        new_sums = set()
        for s in current_sums:
            new_sums.add(s + x)
        new_sums.add(x)
        
        # Check for duplicate sums (union should have size len(current_sums) + len(new_sums))
        all_sums = current_sums.union(new_sums)
        if len(all_sums) != len(current_sums) + len(new_sums):
            return None
        
        # Check divisibility: no two distinct sums in all_sums should divide each other
        # For efficiency, we only need to check pairs where one divides the other
        # Since we're adding new_sums, we need to check:
        # 1. Each s in new_sums against each t in all_sums (s != t)
        # But we can be smarter: for each s in new_sums, check if any t in all_sums \ {s} divides s or is divisible by s
        
        # Convert to list for faster access
        all_sums_list = sorted(all_sums)
        n = len(all_sums_list)
        
        # For each new sum s, check against all other sums
        for s in new_sums:
            for t in all_sums_list:
                if s == t:
                    continue
                if s > t:
                    if s % t == 0:
                        return None
                else:
                    if t % s == 0:
                        return None
        
        return all_sums
    
    # Full validation of a set
    def is_valid_set(elements):
        if not elements:
            return True
        current_sums = set()
        for x in elements:
            new_sums = set()
            for s in current_sums:
                new_sums.add(s + x)
            new_sums.add(x)
            all_sums = current_sums.union(new_sums)
            if len(all_sums) != len(current_sums) + len(new_sums):
                return False
            for s in new_sums:
                for t in all_sums:
                    if s == t:
                        continue
                    if s > t:
                        if s % t == 0:
                            return False
                    else:
                        if t % s == 0:
                            return False
            current_sums = all_sums
        return True
    
    # Greedy construction: add elements one by one, choosing the best candidate
    def greedy_construct(candidates, max_size=None):
        if max_size is None:
            max_size = len(candidates)
        
        current_elements = []
        current_sums = set()
        
        for _ in range(max_size):
            if time.time() > deadline:
                break
            best_candidate = None
            best_sums = None
            
            # Try each candidate
            for x in candidates:
                if x in current_elements:
                    continue
                new_sums = check_addition(current_sums, x)
                if new_sums is not None:
                    # Choose the candidate that gives the "easiest" future
                    # Heuristic: prefer larger numbers (fewer multiples in range)
                    if best_candidate is None or x > best_candidate:
                        best_candidate = x
                        best_sums = new_sums
            
            if best_candidate is None:
                break
            
            current_elements.append(best_candidate)
            current_sums = best_sums
            # Remove best_candidate from candidates to avoid re-adding
            candidates = [c for c in candidates if c != best_candidate]
        
        return current_elements
    
    # DFS with random restarts and limited branching
    def dfs_limited(current_elements, current_sums, remaining_candidates, target_size, deadline):
        if time.time() > deadline:
            return None
        
        if len(current_elements) == target_size:
            return list(current_elements)
        
        if len(remaining_candidates) < target_size - len(current_elements):
            return None
        
        # Limit branching factor
        branch_limit = min(15, len(remaining_candidates))
        
        for i in range(branch_limit):
            x = remaining_candidates[i]
            
            new_sums = check_addition(current_sums, x)
            if new_sums is not None:
                # Shuffle the remaining candidates for randomization
                rem = remaining_candidates[i+1:]
                random.shuffle(rem)
                
                res = dfs_limited(current_elements + [x], new_sums, rem, target_size, deadline)
                if res is not None:
                    return res
        
        return None
    
    # Find a set of given size using multiple strategies
    def find_set(k, time_limit):
        # Strategy 1: Greedy from different starting points
        # Use larger numbers first as they have fewer divisors/multiples in range
        
        # Candidate pools
        pool_large = list(range(256, 128, -1))  # 129-256
        pool_mid = list(range(128, 64, -1))     # 65-128
        pool_small = list(range(64, 32, -1))    # 33-64
        pool_all = list(range(256, 0, -1))
        
        pools = [pool_large, pool_mid, pool_small, pool_all]
        
        for pool in pools:
            if time.time() > time_limit:
                break
            
            # Try greedy with this pool
            result = greedy_construct(list(pool), max_size=k)
            if len(result) == k:
                if is_valid_set(result):
                    return result
        
        # Strategy 2: DFS with random restarts
        rng = random.Random()
        attempts = 0
        max_attempts = 30
        
        while time.time() < time_limit and attempts < max_attempts:
            # Pick a random pool and shuffle
            pool = random.choice([pool_large, pool_mid, pool_small, pool_all])
            candidates = list(pool)
            rng.shuffle(candidates)
            
            # Limit pool size for DFS
            candidates = candidates[:40]
            
            res = dfs_limited([], set(), candidates, k, time_limit)
            if res is not None:
                if is_valid_set(res):
                    return res
            attempts += 1
        
        return None
    
    # Try sizes from 10 down to 1
    best_set = []
    
    for k in range(10, 0, -1):
        if time.time() > deadline:
            break
        
        # Allocate time based on k
        if k >= 9:
            time_alloc = min(20, deadline - time.time())
        elif k == 8:
            time_alloc = min(15, deadline - time.time())
        elif k == 7:
            time_alloc = min(10, deadline - time.time())
        elif k == 6:
            time_alloc = min(7, deadline - time.time())
        elif k == 5:
            time_alloc = min(4, deadline - time.time())
        elif k == 4:
            time_alloc = min(2, deadline - time.time())
        elif k == 3:
            time_alloc = min(1, deadline - time.time())
        else:
            time_alloc = min(0.5, deadline - time.time())
        
        current_deadline = min(deadline, time.time() + time_alloc)
        
        found = find_set(k, current_deadline)
        if found is not None and len(found) == k:
            if is_valid_set(found):
                best_set = found
                break
    
    # Fallback
    if not best_set:
        # Try simple valid sets
        for test_set in [[100, 101, 103], [100, 101], [100, 102], [100, 103], [100], [200, 201], [250, 251], [255, 256]]:
            if is_valid_set(test_set):
                best_set = test_set
                break
    
    if not best_set:
        best_set = [100]
    
    # Ensure all elements are Python int and in valid range
    best_set = [int(x) for x in best_set if 1 <= x <= 256]
    # Remove duplicates while preserving order
    seen = set()
    unique = []
    for x in best_set:
        if x not in seen:
            seen.add(x)
            unique.append(x)
    best_set = unique
    
    return best_set
