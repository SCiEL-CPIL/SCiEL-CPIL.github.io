import time
from fractions import Fraction
import sys
from functools import lru_cache

_START_TIME = None
_TIME_LIMIT = 88.0  # 90s limit, keep 2s buffer

def _time_exceeded():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return (time.time() - _START_TIME) > _TIME_LIMIT

def _subset_sum_possible(target, items):
    """
    Check if 'target' (Fraction) can be formed by a subset sum of 'items' (list of Fractions).
    items are all positive.
    Returns True if possible, False otherwise.
    """
    if target == 0:
        return True
    if not items:
        return False
    if target < 0:
        return False
    
    # Sort items descending to hit large sums quickly and prune
    # We use a tuple of (num, den) for sorting to avoid float issues
    items_sorted = sorted(items, key=lambda x: (x.numerator, -x.denominator), reverse=True)
    
    n = len(items_sorted)
    # Precompute suffix sums for lower bound pruning
    suffix_sums = [Fraction(0)] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sums[i] = suffix_sums[i + 1] + items_sorted[i]
        
    # If total sum < target, impossible
    if suffix_sums[0] < target:
        return False
        
    # If the smallest item is larger than target, impossible (since all are positive)
    # Wait, if target is 1/10 and smallest is 1/6, we can't use 1/6.
    # But we can use larger items? No, if smallest > target, all items > target.
    # So sum of any non-empty subset > target.
    # Only empty subset sums to 0.
    if items_sorted[-1] > target:
        return False

    # Memoization key: target and the sorted tuple of items
    # We convert to a hashable format
    # Using a tuple of (num, den) pairs
    items_key = tuple((item.numerator, item.denominator) for item in items_sorted)
    target_key = (target.numerator, target.denominator)
    
    # We cannot easily lru_cache a nested function that depends on global state or complex objects 
    # unless we pass them as arguments. 
    # However, creating a new function every time is slow.
    # Let's use a global cache dictionary.
    
    global _SUBSET_SUM_CACHE
    cache_key = (target_key, items_key)
    if cache_key in _SUBSET_SUM_CACHE:
        return _SUBSET_SUM_CACHE[cache_key]
        
    def dfs(idx, current_sum):
        if _time_exceeded():
            return False
        if current_sum == target:
            return True
        if idx == n:
            return False
        if current_sum > target:
            return False
        # Prune if even taking all remaining items can't reach target
        if current_sum + suffix_sums[idx] < target:
            return False
            
        # Try taking items[idx]
        if dfs(idx + 1, current_sum + items_sorted[idx]):
            return True
        # Try skipping items[idx]
        if dfs(idx + 1, current_sum):
            return True
        return False
        
    result = dfs(0, Fraction(0))
    _SUBSET_SUM_CACHE[cache_key] = result
    return result

_SUBSET_SUM_CACHE = {}

def _can_add(x, S):
    """
    Check if x can be added to set S (list of ints) without violating constraints.
    S is a list of selected integers.
    """
    if x in S:
        return False
        
    recip_x = Fraction(1, x)
    recips_S = [Fraction(1, y) for y in S]
    
    # Check 1: 1/x is not a subset sum of recips_S
    if _subset_sum_possible(recip_x, recips_S):
        return False
        
    # Check 2: For each y in S, 1/y is not a subset sum of S ∪ {x} \ {y}
    # This is equivalent to: 1/y - 1/x is not a subset sum of S \ {y, x}
    # (If 1/y - 1/x < 0, it's automatically safe since items are positive)
    
    for y in S:
        if y < x:
            # 1/y > 1/x, so target is positive
            target = Fraction(1, y) - recip_x
            # Items are reciprocals of S excluding y and x
            items = [Fraction(1, z) for z in S if z != y and z != x]
            if _subset_sum_possible(target, items):
                return False
        # If y > x, target is negative, safe.
        
    return True

def _greedy_initial():
    """
    Greedy strategy: Add elements from 24 down to 1.
    """
    S = []
    for x in range(24, 0, -1):
        if _can_add(x, S):
            S.append(x)
    return S

def construct():
    global _START_TIME
    _START_TIME = time.time()
    _SUBSET_SUM_CACHE.clear()
    
    # 1. Get a good initial solution
    best_set = _greedy_initial()
    
    # 2. Branch and Bound to find the true maximum
    def search(i, S):
        nonlocal best_set
        if _time_exceeded():
            return
            
        # Pruning: Even if we add all remaining elements, we can't beat best_set
        if len(S) + (24 - i) <= len(best_set):
            return
            
        if i == 24:
            if len(S) > len(best_set):
                best_set = list(S)
            return
            
        x = i + 1
        
        # Strategy: Exclude first for small numbers (they are restrictive)
        # But if we have a lot of room, maybe try Include first?
        # Let's stick to Exclude first to find the "safe" large set quickly,
        # then the B&B will backtrack and try Include branches if needed.
        
        search(i + 1, S)
        
        # Strategy 2: Include x
        if _can_add(x, S):
            S.append(x)
            search(i + 1, S)
            S.pop()
            
    search(0, [])
    
    if not best_set:
        return [24]
        
    return best_set
