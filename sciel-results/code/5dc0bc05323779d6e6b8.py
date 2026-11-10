"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
P = {'n': 256, 'maximum_size': 30}

def _get_sums(S):
    """
    Compute all nonempty subset sums of S.
    Returns a set of sums, or None if any collision is detected.
    """
    sums = set()
    n = len(S)
    # Bitmask approach for small n, or iterative combination
    # Iterative is often faster in Python for n < 20
    current_sums = {0}
    for x in S:
        new_sums = set()
        for s in current_sums:
            ns = s + x
            if ns in sums:
                return None
            if ns in new_sums:
                return None
            new_sums.add(ns)
        # Check if any new sum collides with existing sums
        # Actually, the check above `if ns in sums` handles collision with old sums.
        # `if ns in new_sums` handles collision within the new batch?
        # Wait, if s1 + x == s2 + x, then s1 == s2, which is impossible since current_sums is a set.
        # So new_sums elements are distinct from each other.
        # We just need to ensure they don't collide with `sums`.
        if new_sums & sums:
            return None
        sums.update(new_sums)
        current_sums = new_sums | current_sums # Actually we just need the full set of sums so far for the next step?
        # No, the standard way:
        # sums_so_far = {0}
        # for x in S:
        #   new_sums = {s + x for s in sums_so_far}
        #   if new_sums & sums_so_far: return None # Collision
        #   sums_so_far = sums_so_far | new_sums
        # This ensures all subset sums are distinct.
        
    # Let's re-implement cleanly
    sums_so_far = {0}
    for x in S:
        new_sums = set()
        for s in sums_so_far:
            ns = s + x
            if ns in sums_so_far:
                return None
            new_sums.add(ns)
        sums_so_far = sums_so_far | new_sums
    return sums_so_far - {0} # Exclude empty sum

def _check_divisibility(sums_set):
    """
    Check if no two distinct values in sums_set divide one another.
    """
    if not sums_set:
        return True
    sums_list = sorted(sums_set)
    sum_set = set(sums_list)
    max_sum = sums_list[-1]
    for s in sums_list:
        # Check multiples of s
        multiple = s * 2
        while multiple <= max_sum:
            if multiple in sum_set:
                return False
            multiple += s
    return True

def _is_valid_set(S):
    """
    Check if S is a valid set.
    """
    if len(set(S)) != len(S):
        return False
    if any(not isinstance(x, int) or isinstance(x, bool) or x < 1 or x > 256 for x in S):
        return False
    sums = _get_sums(S)
    if sums is None:
        return False
    return _check_divisibility(sums)

def _try_add(S, x):
    """
    Try to add x to S. Returns True if S + [x] is valid.
    """
    if x in S:
        return False
    new_S = S + [x]
    return _is_valid_set(new_S)

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    best_set = []
    
    # Strategy 1: Greedy construction
    # We try to build a set by adding elements one by one.
    # We iterate through candidates in a specific order.
    # Smaller numbers create more sums, which might be bad for divisibility.
    # Larger numbers are "safer" but might not fit as many.
    # Let's try increasing order first.
    
    current_set = []
    for x in range(1, 257):
        if time.time() > deadline - 10:
            break
        if _try_add(current_set, x):
            current_set.append(x)
            if len(current_set) > len(best_set):
                best_set = list(current_set)
    
    # Strategy 2: Local Search / Hill Climbing
    # Try to swap elements or replace elements to increase size or maintain size with better properties.
    # Since we want to maximize size, we can try to find a set of size k that is valid.
    # We can use random sampling of sets of size k.
    
    # Let's try to improve the greedy result by random restarts of greedy with different orders.
    # Or random search for larger sets.
    
    # If greedy found a small set, try random sets of increasing size.
    current_best_size = len(best_set)
    
    # Random search for sets of size k
    # Start from k = current_best_size + 1
    # We have limited time, so we can't search huge spaces.
    # However, n=256, k=10 is C(256,10) which is huge.
    # But valid sets are rare.
    
    # Better: Use the greedy result as a starting point for local search.
    # Try to replace an element with a larger one to free up "divisibility space"?
    # Or try to add more elements by removing one?
    
    # Let's try a simple randomized greedy.
    # Shuffle the candidate list, then greedily add.
    
    for _ in range(200): # 200 restarts
        if time.time() > deadline - 5:
            break
        
        candidates = list(range(1, 257))
        random.shuffle(candidates)
        
        curr = []
        for x in candidates:
            if time.time() > deadline - 5:
                break
            if _try_add(curr, x):
                curr.append(x)
                if len(curr) > len(best_set):
                    best_set = list(curr)
                    current_best_size = len(best_set)
        
        # If we found a larger set, we can break early or continue to find even larger
        if current_best_size >= 15: # Heuristic upper bound guess
            break
            
    # Final validation and return
    if _is_valid_set(best_set):
        return list(best_set)
    else:
        # Fallback to known valid set
        return [31, 32, 34]

# EVOLVE-BLOCK-END
