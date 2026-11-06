"""Program for large reciprocal-subsum-free sets using optimized greedy construction."""
import time
import math

_START_TIME = time.time()
_DEADLINE = 85.0  # Safety margin under the 90s wall-clock limit


def _time_remaining():
    return _DEADLINE - (time.time() - _START_TIME)


def _lcm(a, b):
    return a * b // math.gcd(a, b)


def _can_form_sum(target, items):
    """
    Check if target can be formed as a sum of a subset of items.
    Uses meet-in-the-middle for efficiency.
    """
    n = len(items)
    if n == 0:
        return target == 0
    if target == 0:
        return True
    if sum(items) < target:
        return False
    
    # Sort items descending to potentially prune earlier if we used DFS, 
    # but for MITM it doesn't matter much for generation.
    mid = n // 2
    left = items[:mid]
    right = items[mid:]
    
    # Generate all subset sums for left half
    left_sums = {0}
    for x in left:
        new_sums = set()
        for s in left_sums:
            ns = s + x
            if ns <= target:
                new_sums.add(ns)
        left_sums.update(new_sums)
        
    # Generate all subset sums for right half
    right_sums = {0}
    for x in right:
        new_sums = set()
        for s in right_sums:
            ns = s + x
            if ns <= target:
                new_sums.add(ns)
        right_sums.update(new_sums)
        
    # Check if any pair sums to target
    for s in left_sums:
        if (target - s) in right_sums:
            return True
            
    return False


def _is_subsum_free(S):
    """
    Check if set S is reciprocal-subsum-free.
    For each x in S, 1/x must NOT equal sum(1/y for y in T)
    where T is a non-empty subset of S \\ {x}.
    """
    if len(S) < 2:
        return True
    
    S_list = list(S)
    n = len(S_list)
    
    # Calculate LCM of all elements in S
    L = 1
    for x in S_list:
        L = _lcm(L, x)
        
    # Convert reciprocals to integers: 1/x becomes L/x
    vals = [L // x for x in S_list]
    
    for i in range(n):
        target = vals[i]
        # Items available are all vals except vals[i]
        others = [v for j, v in enumerate(vals) if j != i]
        
        # Check if target can be formed by subset of others
        if _can_form_sum(target, others):
            return False
            
    return True


def construct():
    """
    Return a nonempty subset of [1,24] that is reciprocal-subsum-free,
    maximizing cardinality.
    """
    current = set()
    
    # Greedy descending: try to add elements from 24 down to 1
    # Large numbers have small reciprocals, which are "safer" in terms of 
    # not being formed by sums of other reciprocals? 
    # Actually, we want to avoid 1/x = sum(1/y). 
    # If we pick large numbers first, their reciprocals are small.
    # The sum of other reciprocals will be larger if we pick small numbers later.
    # So picking large numbers first is a good heuristic.
    
    for k in range(24, 0, -1):
        if _time_remaining() < 5.0:
            break
        
        new_set = current | {k}
        if _is_subsum_free(new_set):
            current = new_set
            
    # If we have time, try to improve by swapping or local search?
    # Given the speed of _is_subsum_free, we can afford some local search.
    # But greedy descending is usually very strong for this type of problem.
    
    # Let's try a simple local search: try to add any missing number
    # by removing one existing number if necessary.
    # This is complex. Let's stick to greedy + maybe a second pass with different order?
    # Or just return the greedy result.
    
    # Let's verify if we can add any number we skipped by removing a conflict?
    # For simplicity and speed, we will just return the greedy set.
    # The previous program found ~20 elements. Greedy should find similar or better.
    
    if not current:
        return [24]
        
    return list(current)
