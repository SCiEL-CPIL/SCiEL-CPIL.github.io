import time
from fractions import Fraction
import itertools

_START_TIME = None
_TIME_LIMIT = 85.0  # 90s limit, keep 5s buffer

def _time_exceeded():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return (time.time() - _START_TIME) > _TIME_LIMIT

# Precompute reciprocals for 1..24
_RECIPS = {i: Fraction(1, i) for i in range(1, 25)}

def _subset_sum_possible_mim(target, items):
    """
    Check if 'target' (Fraction) can be formed by a subset sum of 'items' (list of Fractions).
    Uses Meet-in-the-Middle for efficiency.
    """
    if target == 0:
        return True
    if not items:
        return False
    if target < 0:
        return False
    
    n = len(items)
    if n == 0:
        return False
        
    # If n is small, brute force might be faster due to overhead
    if n <= 10:
        # Brute force
        for mask in range(1, 1 << n):
            s = Fraction(0)
            for i in range(n):
                if mask & (1 << i):
                    s += items[i]
            if s == target:
                return True
        return False
        
    # Meet in the middle
    half = n // 2
    left_items = items[:half]
    right_items = items[half:]
    
    # Generate all subset sums for left half
    left_sums = set()
    left_sums.add(Fraction(0))
    for i in range(len(left_items)):
        # To avoid modifying the set while iterating, we collect new sums
        new_sums = []
        for s in left_sums:
            new_sums.append(s + left_items[i])
        left_sums.update(new_sums)
        
    # Generate all subset sums for right half and check for target
    right_sums = set()
    right_sums.add(Fraction(0))
    for i in range(len(right_items)):
        new_sums = []
        for s in right_sums:
            new_sums.append(s + right_items[i])
        right_sums.update(new_sums)
        
    # Check if any pair (l, r) such that l + r == target
    # Since we have left_sums and right_sums, we can check:
    # For each l in left_sums, is (target - l) in right_sums?
    for l in left_sums:
        if (target - l) in right_sums:
            return True
            
    return False

def _can_add(x, S):
    """
    Check if x can be added to set S (list of ints) without violating constraints.
    """
    if x in S:
        return False
        
    recip_x = _RECIPS[x]
    
    # Check 1: 1/x is not a subset sum of reciprocals of S
    if S:
        recips_S = [_RECIPS[y] for y in S]
        if _subset_sum_possible_mim(recip_x, recips_S):
            return False
        
        # Check 2: For each y in S, 1/y is not a subset sum of S ∪ {x} \ {y}
        # This is equivalent to: 1/y - 1/x is not a subset sum of S \ {y, x}
        # Note: If 1/y < 1/x (y > x), then 1/y - 1/x < 0, so it's impossible for positive items.
        # So we only need to check for y < x.
        for y in S:
            if y < x:
                target = _RECIPS[y] - recip_x
                if target <= 0:
                    continue
                # Items are reciprocals of S excluding y and x
                items = [_RECIPS[z] for z in S if z != y and z != x]
                if _subset_sum_possible_mim(target, items):
                    return False
            # If y > x, target is negative, safe.
            
    return True

def _greedy_initial():
    """
    Greedy strategy: Try multiple orders to find a good initial set.
    """
    best_S = []
    orders = [
        list(range(24, 0, -1)),
        list(range(1, 25)),
        list(range(12, 0, -1)) + list(range(24, 12, -1)), # Mixed
        list(range(24, 12, -1)) + list(range(12, 0, -1)), # Large first
    ]
    
    for order in orders:
        S = []
        for x in order:
            if _time_exceeded():
                break
            if _can_add(x, S):
                S.append(x)
        if len(S) > len(best_S):
            best_S = S[:]
            
    return best_S

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    # 1. Get a good initial solution
    best_set = _greedy_initial()
    
    # 2. Branch and Bound
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
        
        # Strategy: Try Include first if x is large (safer), Exclude first if x is small
        if x > 12:
            if _can_add(x, S):
                S.append(x)
                search(i + 1, S)
                S.pop()
            search(i + 1, S)
        else:
            search(i + 1, S)
            if _can_add(x, S):
                S.append(x)
                search(i + 1, S)
                S.pop()
            
    search(0, [])
    
    if not best_set:
        return [24]
        
    return best_set
