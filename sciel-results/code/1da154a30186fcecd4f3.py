import time
import sys

_LCM = 232792560

def _verify_recip_dissociated(subset):
    if not subset:
        return True
    k = len(subset)
    values = [_LCM // i for i in subset]
    sums = {0}
    for v in values:
        new_sums = {s + v for s in sums}
        if sums & new_sums:
            return False
        sums |= new_sums
    return True

def _can_add(sums, v):
    # Check if adding v causes collision
    # Collision if (s + v) in sums for any s in sums
    shifted = {s + v for s in sums}
    return not (sums & shifted)

def _backtrack_search(K, start_idx, current_set, current_sums, deadline, order, values):
    """
    Recursively try to add elements from 'order' starting at 'start_idx'
    to reach size K.
    """
    if len(current_set) == K:
        return current_set[:]
    
    if time.time() > deadline:
        return None
        
    # Pruning: not enough elements left
    remaining = len(order) - start_idx
    if len(current_set) + remaining < K:
        return None

    need = K - len(current_set)
    n = len(order)
    
    for i in range(start_idx, n):
        # Pruning: not enough elements left to reach K
        if len(current_set) + (n - i) < K:
            break
            
        idx = order[i]
        v = values[i]
        
        # Check if we can add idx
        if _can_add(current_sums, v):
            # Add it
            new_sums = set(current_sums)
            for s in current_sums:
                new_sums.add(s + v)
            
            # Recurse
            result = _backtrack_search(K, i + 1, current_set + [idx], new_sums, deadline, order, values)
            if result:
                return result
            # If no result, try next (backtrack)
        # If cannot add, skip (try next)
        
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    candidates = list(range(1, 23))
    
    # 1. Greedy Baseline
    def greedy(order):
        current_set = []
        current_sums = {0}
        for idx in order:
            v = _LCM // idx
            if _can_add(current_sums, v):
                current_set.append(idx)
                for s in list(current_sums):
                    current_sums.add(s + v)
        return current_set

    # Try a few greedy orderings to get a good baseline
    best_greedy = []
    best_greedy_size = 0
    
    # Order 1: Ascending
    s1 = greedy(candidates)
    if len(s1) > best_greedy_size:
        best_greedy = s1
        best_greedy_size = len(s1)
        
    # Order 2: Descending
    s2 = greedy(sorted(candidates, reverse=True))
    if len(s2) > best_greedy_size:
        best_greedy = s2
        best_greedy_size = len(s2)

    # Order 3: Random (fixed seed)
    import random
    rng = random.Random(42)
    for _ in range(20):
        shuffled = candidates[:]
        rng.shuffle(shuffled)
        s3 = greedy(shuffled)
        if len(s3) > best_greedy_size:
            best_greedy = s3
            best_greedy_size = len(s3)

    # Current best is likely 16.
    # Now try to find size 17 using backtracking.
    
    target_size = best_greedy_size + 1
    if target_size > 22:
        target_size = 22
        
    found = None
    
    search_deadline = start_time + 45.0 # Give 45s to backtracking
    
    # Try multiple orderings for backtracking
    orderings = [
        candidates,  # ascending
        sorted(candidates, reverse=True),  # descending
    ]
    
    # Add some random orderings
    rng2 = random.Random(123)
    for _ in range(5):
        shuffled = candidates[:]
        rng2.shuffle(shuffled)
        orderings.append(shuffled)
    
    for order in orderings:
        if time.time() > search_deadline:
            break
        values_order = [_LCM // i for i in order]
        res = _backtrack_search(target_size, 0, [], {0}, search_deadline, order, values_order)
        if res:
            found = res
            break

    if found:
        return found
    else:
        return best_greedy
