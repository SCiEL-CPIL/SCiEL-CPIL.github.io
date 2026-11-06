import time
from fractions import Fraction
from itertools import combinations

_START = None
DEADLINE = 55.0

def _deadline_hit():
    global _START
    if _START is None:
        return False
    return (time.time() - _START) > DEADLINE

def _generate_subset_sums(elements):
    """
    Generate all possible subset sums of reciprocals of 'elements'.
    Returns a set of Fractions.
    Includes the empty sum (0).
    """
    sums = {Fraction(0, 1)}
    for e in elements:
        new_sums = set()
        rec = Fraction(1, e)
        for s in sums:
            new_sums.add(s + rec)
        sums.update(new_sums)
    return sums

def _is_subset_sum(target, elements):
    """
    Check if 'target' is a subset sum of reciprocals of 'elements'.
    Uses meet-in-the-middle for larger sets.
    """
    if target <= 0:
        return False
    if not elements:
        return target == 0
    
    n = len(elements)
    if n <= 12:
        # Direct generation is fast enough for small n
        sums = _generate_subset_sums(elements)
        return target in sums
    else:
        # Meet in the middle
        half = n // 2
        left = elements[:half]
        right = elements[half:]
        
        left_sums = _generate_subset_sums(left)
        right_sums = _generate_subset_sums(right)
        
        # Check if any l + r == target => r == target - l
        # Iterate over smaller set for efficiency if needed, but both are ~2^6
        for l in left_sums:
            if (target - l) in right_sums:
                return True
        return False

def _can_add(S, v):
    """
    Check if v can be added to the valid set S.
    S is a list of integers.
    """
    # 1. Check if 1/v is a subset sum of reciprocals of S
    #    i.e., exists T subset of S such that sum(1/t for t in T) == 1/v
    if _is_subset_sum(Fraction(1, v), S):
        return False
        
    # 2. Check if adding v creates a violation for existing x in S
    #    Violation: 1/x == sum(1/t for t in T) where T subset of S U {v} \ {x}
    #    Since S is valid, any such T must contain v.
    #    So 1/x == 1/v + sum(1/t for t in T') where T' subset of S \ {x}
    #    => 1/x - 1/v == sum(1/t for t in T')
    #    This requires 1/x > 1/v => x < v.
    #    So we only check x in S such that x < v.
    
    for x in S:
        if x < v:
            target = Fraction(1, x) - Fraction(1, v)
            # Check if target is a subset sum of reciprocals of S \ {x}
            # Note: S \ {x} does not contain v, so we just use S with x removed
            others = [u for u in S if u != x]
            if _is_subset_sum(target, others):
                return False
                
    return True

def _solve():
    global _START
    _START = time.time()
    
    candidates = list(range(1, 25))
    best_set = []
    best_size = 0
    
    def backtrack(idx, S):
        nonlocal best_set, best_size
        
        if _deadline_hit():
            return
            
        # Pruning: max possible size
        if len(S) + (len(candidates) - idx) <= best_size:
            return
            
        if idx == len(candidates):
            if len(S) > best_size:
                best_size = len(S)
                best_set = S[:]
            return
            
        v = candidates[idx]
        
        # Option 1: Include v
        if _can_add(S, v):
            S.append(v)
            backtrack(idx + 1, S)
            S.pop()
            
        # Option 2: Exclude v
        backtrack(idx + 1, S)
        
    backtrack(0, [])
    
    return best_set

def construct():
    result = _solve()
    if not result:
        result = [24]
    return result
