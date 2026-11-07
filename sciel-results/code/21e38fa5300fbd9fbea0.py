import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin
    
    # Heuristic: Try to find a set of size K
    # Max possible size? 
    # If we pick 1, we can't pick anything else.
    # If we pick 2, 3: sums {2,3,5}. 2|? no. 3|? no. 5|? no.
    # If we pick 2, 5: sums {2,5,7}.
    # Let's just search for max size.
    
    best_set = []
    
    # Candidate pool
    # We can restrict candidates to [2, 256] since 1 is problematic for size > 1.
    # Actually, if 1 is in the set, size is 1. So for size > 1, 1 is excluded.
    
    def is_valid(sums, x):
        # sums is a set of current subset sums
        # x is the new element
        # New sums: sums U {x} U {s+x for s in sums}
        
        # Check collision first
        new_sums = set(sums)
        new_sums.add(x)
        for s in sums:
            new_sums.add(s + x)
        
        if len(new_sums) != len(sums) + 1 + len(sums):
            return None, False
            
        # Check divisibility
        # We need to check all pairs in new_sums
        # Optimization: if a|b and a!=b, invalid.
        # We can sort and check?
        # Or just double loop.
        lst = list(new_sums)
        n = len(lst)
        for i in range(n):
            a = lst[i]
            for j in range(n):
                if i == j: continue
                b = lst[j]
                if b % a == 0:
                    return None, False
        return new_sums, True

    def dfs(current_set, current_sums, depth, target_size, candidates):
        nonlocal best_set
        if time.time() > deadline:
            return False
        
        if len(current_set) == target_size:
            best_set = list(current_set)
            return True
            
        if not candidates:
            return False
            
        # Try each candidate
        # Randomize order to escape local minima
        random.shuffle(candidates)
        
        for i in range(len(candidates)):
            x = candidates[i]
            new_sums, valid = is_valid(current_sums, x)
            if valid:
                # Recurse
                # Create new candidate list excluding x
                # To save time, we can just pass a set or list
                new_candidates = [c for c in candidates if c != x]
                
                if dfs(current_set + [x], new_sums, depth + 1, target_size, new_candidates):
                    return True
                    
        return False

    # Search for maximum size
    # Start from a reasonable upper bound
    # With 256 max value, and distinct sums, size is limited.
    # 2^k sums < 256 * k ? No.
    # Max sum is sum of elements.
    # Let's try sizes 10 down to 1.
    
    # Initial candidates: 2..256
    # We can prune candidates that are multiples of existing sums?
    
    # To increase chance of finding a large set, we might want to bias towards
    # numbers that are "large" or "prime" to avoid divisibility.
    
    for target_size in range(12, 0, -1):
        if time.time() > deadline:
            break
            
        # We can run multiple random restarts for each target size
        # Or just one deep DFS with randomized order.
        
        # Let's try a few restarts
        found = False
        for restart in range(5):
            if time.time() > deadline:
                break
            candidates = list(range(2, 257))
            # Shuffle candidates initially?
            # The DFS shuffles at each level.
            
            if dfs([], set(), 0, target_size, candidates):
                found = True
                break
                
        if found:
            break
            # We found a set of size target_size. 
            # We could try to extend it? No, we iterate down from max.
            # So the first found is the max.
            
    if not best_set:
        # Fallback: return a small valid set
        best_set = [2, 3] # Sums: 2,3,5. 2|? no. 3|? no. 5|? no. Valid.
        
    # Ensure types are correct
    return [int(x) for x in best_set]
