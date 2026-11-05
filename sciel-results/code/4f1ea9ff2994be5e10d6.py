"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from functools import reduce

def _lcm(a, b):
    return a * b // gcd(a, b)

def _lcm_list(lst):
    if not lst:
        return 1
    return reduce(_lcm, lst)

def construct():
    start_time = time.time()
    deadline = start_time + 70  # safety margin before 90s kill
    
    # Greedy: try elements from 24 down to 1
    # When adding element x, check if 1/x can be formed by subset of reciprocals of already-selected elements
    # Already-selected elements are all > x, so their reciprocals are all < 1/x
    
    selected = []  # list of selected elements (will be in descending order)
    
    for x in range(24, 0, -1):
        if time.time() > deadline:
            break
        
        # Check: can 1/x be formed as sum of reciprocals of a non-empty subset of 'selected'?
        # Convert to integer arithmetic: multiply everything by LCM of all denominators
        if not selected:
            selected.append(x)
            continue
        
        # Compute LCM of all elements in selected
        L = _lcm_list(selected)
        
        # Target: 1/x * L = L // x (must be integer since we'll check divisibility)
        # Actually, 1/x as fraction: we need sum of (L//y) for y in subset to equal L//x
        # But L//x might not be integer if x doesn't divide L. Let's use a common denominator approach.
        
        # Better: use LCM of selected and x
        L2 = _lcm(L, x)
        
        # Target in integer terms: L2 // x
        target = L2 // x
        
        # Each selected element y contributes L2 // y
        # We need to check if any non-empty subset sums to target
        values = [L2 // y for y in selected]
        
        # Subset sum check with meet-in-the-middle or DFS
        # Since selected size is at most 24, but in practice greedy will keep it small
        # Use DFS with memoization on (index, remaining)
        
        n = len(values)
        # Sort values in descending order for better pruning
        values_sorted = sorted(values, reverse=True)
        
        # Quick check: if sum of all < target, impossible
        if sum(values_sorted) < target:
            selected.append(x)
            continue
        
        # If any single value equals target, invalid
        if target in values_sorted:
            # 1/x equals 1/y for some y in selected, meaning x == y, but x < all in selected
            # Actually this can't happen since x < y means 1/x > 1/y, so L2//x > L2//y
            # So target > all values, can't be a single element
            pass
        
        # DFS: can we form target from subset of values_sorted?
        # Use recursive with pruning
        found = [False]
        
        def dfs(idx, remaining):
            if found[0]:
                return
            if remaining == 0:
                found[0] = True
                return
            if idx == n:
                return
            if remaining < 0:
                return
            # Pruning: if sum of remaining elements < remaining, impossible
            # Precompute suffix sums
            if _suffix_sums[idx] < remaining:
                return
            
            # Option 1: take values_sorted[idx]
            dfs(idx + 1, remaining - values_sorted[idx])
            # Option 2: skip
            dfs(idx + 1, remaining)
        
        # Precompute suffix sums for pruning
        _suffix_sums = [0] * (n + 1)
        for i in range(n - 1, -1, -1):
            _suffix_sums[i] = _suffix_sums[i + 1] + values_sorted[i]
        
        dfs(0, target)
        
        if not found[0]:
            selected.append(x)
    
    # Return as a list (order doesn't matter for the contract)
    if not selected:
        return [24]
    return list(selected)

# EVOLVE-BLOCK-END
