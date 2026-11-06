"""Rewritten program for the reciprocal-subsum-free set task."""
import time
from fractions import Fraction
from itertools import combinations

_START = None
DEADLINE = 55.0 # Safety margin

def _deadline_hit():
    global _START
    if _START is None:
        return False
    return (time.time() - _START) > DEADLINE

def _subset_sum_check(target, candidates):
    """
    Check if 'target' (Fraction) can be formed by a subset sum of reciprocals of 'candidates'.
    'candidates' is a list of integers.
    """
    if target <= 0:
        return False
    if not candidates:
        return False
    
    n = len(candidates)
    
    if n <= 15:
        for mask in range(1, 1 << n):
            s = Fraction(0, 1)
            for i in range(n):
                if mask & (1 << i):
                    s += Fraction(1, candidates[i])
                    if s > target:
                        break
            else:
                if s == target:
                    return True
        return False
    else:
        half = n // 2
        left = candidates[:half]
        right = candidates[half:]
        
        left_sums = set()
        for mask in range(1 << len(left)):
            s = Fraction(0, 1)
            for i in range(len(left)):
                if mask & (1 << i):
                    s += Fraction(1, left[i])
            left_sums.add(s)
            
        for mask in range(1 << len(right)):
            s = Fraction(0, 1)
            for i in range(len(right)):
                if mask & (1 << i):
                    s += Fraction(1, right[i])
            if (target - s) in left_sums:
                return True
        return False

def _is_valid_addition(S, v):
    """
    Check if adding v to S (where all u in S < v) maintains validity.
    S is a list of selected integers.
    """
    for u in S:
        target = Fraction(1, u) - Fraction(1, v)
        if target <= 0:
            continue
        
        pool = [w for w in S if w > u]
        
        if not pool:
            continue
            
        total_sum = sum(Fraction(1, w) for w in pool)
        if total_sum < target:
            continue
            
        if _subset_sum_check(target, pool):
            return False
            
    return True

def _greedy_construct():
    """
    Greedily construct a large set by adding elements from largest to smallest.
    """
    S = []
    for v in range(24, 0, -1):
        if _is_valid_addition(S, v):
            S.append(v)
    return S

def _solve():
    """
    Backtracking search to find the maximum size valid subset of [1, 24].
    """
    candidates = list(range(24, 0, -1)) # Descending order
    best_set = []
    best_size = 0
    
    # Initialize with greedy result
    greedy_set = _greedy_construct()
    if len(greedy_set) > best_size:
        best_size = len(greedy_set)
        best_set = greedy_set[:]
        
    def backtrack(idx, S):
        nonlocal best_set, best_size
        
        if _deadline_hit():
            return
            
        if len(S) + (len(candidates) - idx) <= best_size:
            return
            
        if idx == len(candidates):
            if len(S) > best_size:
                best_size = len(S)
                best_set = S[:]
            return
            
        # Option 1: Exclude candidates[idx]
        backtrack(idx + 1, S)
        
        # Option 2: Include candidates[idx]
        v = candidates[idx]
        
        # Note: S is not necessarily sorted in ascending order for _is_valid_addition
        # _is_valid_addition assumes S contains elements < v?
        # In the original code, S was built in ascending order.
        # Here we build in descending order.
        # So S contains elements > v.
        # We need to check if adding v is valid.
        # The condition is symmetric?
        # "no reciprocal of a selected element may equal the sum of reciprocals of a subset of the OTHER selected elements"
        # If we add v, we must check:
        # 1. 1/v is not sum of subset of S.
        # 2. For each u in S, 1/u is not sum of subset of (S \ {u}) U {v}.
        #    This is equivalent to: 1/u - 1/v is not sum of subset of (S \ {u, v})?
        #    No, 1/u = sum(T) where T subset of (S \ {u}) U {v}.
        #    If v not in T, then 1/u is sum of subset of S \ {u}. (Already valid in S).
        #    If v in T, then 1/u = 1/v + sum(T') where T' subset of S \ {u, v}.
        #    So 1/u - 1/v = sum(T').
        #    So we need to check if 1/u - 1/v is a subset sum of S \ {u, v}.
        
        # So _is_valid_addition(S, v) needs to be updated for S containing larger elements.
        # Let's write a specific check for this case.
        
        # Check 1: 1/v not sum of subset of S
        if _subset_sum_check(Fraction(1, v), S):
            # Cannot add v
            pass
        else:
            # Check 2: For each u in S, 1/u - 1/v not sum of subset of S \ {u}
            valid = True
            for u in S:
                target = Fraction(1, u) - Fraction(1, v)
                if target <= 0:
                    continue
                pool = [w for w in S if w != u]
                if _subset_sum_check(target, pool):
                    valid = False
                    break
            
            if valid:
                S.append(v)
                backtrack(idx + 1, S)
                S.pop()
            
    backtrack(0, [])
    
    return best_set

def construct():
    global _START
    _START = time.time()
    
    result = _solve()
    
    if not result:
        result = [24]
        
    return result
