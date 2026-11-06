"""Rewritten program for the reciprocal-subsum-free set task."""
import time
from fractions import Fraction
from itertools import combinations

_START = None
DEADLINE = 75.0

def _deadline_hit():
    global _START
    if _START is None:
        return False
    return (time.time() - _START) > DEADLINE

def _check_violation(S, a):
    """Check if 1/a equals sum of reciprocals of some subset of S \\ {a}.
    Returns True if a violation exists (i.e., the constraint is broken)."""
    others = [x for x in S if x != a]
    target = Fraction(1, a)
    # Subset sum over reciprocals of others
    # Meet-in-the-middle for efficiency if needed, but for |others| <= 23, 
    # we can do a DFS with pruning
    n = len(others)
    if n == 0:
        return False
    
    # Use meet-in-the-middle for n > 15
    if n <= 15:
        # Brute force all subsets
        for mask in range(1, 1 << n):
            s = Fraction(0, 1)
            for i in range(n):
                if mask & (1 << i):
                    s += Fraction(1, others[i])
            if s == target:
                return True
        return False
    else:
        # Meet in the middle
        half = n // 2
        left = others[:half]
        right = others[half:]
        
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

def _is_valid(S):
    """Check if S is a valid reciprocal-subsum-free set."""
    S_list = list(S)
    for a in S_list:
        if _check_violation(S_list, a):
            return False
    return True

def _greedy_construct():
    """Greedy construction: try to build a large valid set."""
    # Start with all elements and remove problematic ones
    # Try multiple strategies
    
    best = list(range(13, 25))  # seed
    
    # Strategy 1: Start with all [1,24], remove elements that cause violations
    S = list(range(1, 25))
    changed = True
    while changed and not _deadline_hit():
        changed = False
        for a in sorted(S):  # remove smallest first (most constrained)
            if _check_violation(S, a):
                S.remove(a)
                changed = True
                break
    
    if len(S) > len(best):
        best = S
    
    # Strategy 2: Greedy add in various orders
    for order in [range(24, 0, -1), range(1, 25), range(12, 0, -1), range(13, 25)]:
        S = []
        for a in order:
            if _deadline_hit():
                break
            new_S = S + [a]
            if _is_valid(new_S):
                S = new_S
        if len(S) > len(best):
            best = S
    
    # Strategy 3: Start from seed, try to add elements
    S = list(range(13, 25))
    for a in range(12, 0, -1):
        if _deadline_hit():
            break
        new_S = S + [a]
        if _is_valid(new_S):
            S = new_S
    if len(S) > len(best):
        best = S
    
    return best

def construct():
    global _START
    _START = time.time()
    
    result = _greedy_construct()
    
    # Safety: ensure nonempty
    if not result:
        result = [24]
    
    return result
