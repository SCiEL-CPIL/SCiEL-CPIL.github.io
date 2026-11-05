import time
from math import gcd
from functools import reduce
from itertools import combinations

def _lcm(a, b):
    return a * b // gcd(a, b)

def _lcm_list(lst):
    if not lst:
        return 1
    return reduce(_lcm, lst)

def _subset_sum_exists(values, target):
    """
    Check if any non-empty subset of values sums to target.
    Uses meet-in-the-middle for efficiency.
    """
    if target < 0:
        return False
    if not values:
        return False
    if target in values:
        return True
    if sum(values) < target:
        return False
    
    n = len(values)
    # If n is small, simple bitset or set of sums might be faster than MITM overhead?
    # MITM is O(2^(n/2)). For n=10, 2^5=32. Very fast.
    
    mid = n // 2
    left = values[:mid]
    right = values[mid:]
    
    left_sums = set()
    # Generate all subset sums for left
    for mask in range(1 << len(left)):
        s = 0
        for i in range(len(left)):
            if mask & (1 << i):
                s += left[i]
        left_sums.add(s)
        
    # Check right subsets
    for mask in range(1 << len(right)):
        s = 0
        for i in range(len(right)):
            if mask & (1 << i):
                s += right[i]
        if (target - s) in left_sums:
            # Ensure we don't pick empty subset from both sides
            # If mask==0 and target-s==0, then target==0.
            # But target > 0 (since x < L).
            # So if target > 0, we are safe.
            return True
            
    return False

def _is_valid_set(selected):
    """
    Check if the set is reciprocal-subsum-free.
    """
    for x in selected:
        others = [y for y in selected if y != x]
        if not others:
            continue
        L = _lcm_list(others)
        target = L // x
        values = [L // y for y in others]
        if _subset_sum_exists(values, target):
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    best_set = []
    
    # We search for sets in decreasing order of elements.
    # We use backtracking.
    # To speed up, we can pre-check pairs or triples?
    # Let's just do DFS.
    
    # Optimization: Sort candidates in decreasing order.
    candidates = list(range(24, 0, -1))
    
    def dfs(current_set, idx):
        nonlocal best_set
        if time.time() > deadline:
            return
        
        if len(current_set) > len(best_set):
            best_set = list(current_set)
            
        # Pruning: If even adding all remaining candidates doesn't beat best, stop?
        # Max possible size = len(current_set) + (24 - idx)
        # This is weak.
        
        for i in range(idx, 24):
            x = candidates[i]
            # Try adding x
            new_set = current_set + [x]
            
            # Check validity of new_set
            # We only need to check if adding x breaks validity.
            # 1. Does x form a sum from current_set?
            # 2. Does any y in current_set form a sum from (current_set \ {y}) U {x}?
            
            # Check 1:
            if current_set:
                L = _lcm_list(current_set)
                target_x = L // x
                values_x = [L // y for y in current_set]
                if _subset_sum_exists(values_x, target_x):
                    continue # x cannot be added
            
            # Check 2: For each y in current_set, check if 1/y is formed by others.
            # This is expensive.
            # However, if current_set was valid, adding x only creates new sums involving x.
            # So we only need to check if 1/y = sum_{z in T} 1/z where x in T.
            # This implies 1/y = 1/x + sum_{z in T'} 1/z where T' subset of current_set \ {y}.
            
            valid = True
            for y in current_set:
                others_y = [z for z in current_set if z != y]
                # We need to check if L_new / y is a subset sum of {L_new / z for z in (others_y + [x])}
                # L_new = lcm(current_set + [x])
                # This is getting heavy.
                
                # Alternative: Just call _is_valid_set(new_set)
                # For n=10, _is_valid_set does 10 subset sum checks.
                # Each subset sum check is O(2^5) = 32 ops.
                # Total ~320 ops per validity check.
                # DFS nodes: 2^24 is too many.
                # But we prune heavily.
                
                # Let's try a heuristic:
                # If x is large, 1/x is small.
                # If we add x, it's unlikely to break validity of large y's.
                # It's most likely to break validity of small y's (large reciprocals).
                
                # Let's just use the full check for now, it might be fast enough for N=24 if pruning is good.
                # Actually, let's optimize the check.
                
            # Let's use the full _is_valid_set for simplicity first, with a time check.
            if _is_valid_set(new_set):
                dfs(new_set, i + 1)
            
            if time.time() > deadline:
                return
                
    dfs([], 0)
    
    if not best_set:
        return [24]
        
    return best_set
