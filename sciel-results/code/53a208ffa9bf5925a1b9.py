from fractions import Fraction
import time
import itertools
from math import gcd

_DEADLINE = None
_BEST = None

def _lcm(a, b):
    return a * b // gcd(a, b)

def _subset_sum_exists_int(target_int, elements_int):
    """
    Check if target_int can be formed by a subset of elements_int using meet-in-the-middle.
    elements_int are positive integers.
    """
    if not elements_int:
        return False
    elements_int = sorted(elements_int, reverse=True)
    total = sum(elements_int)
    if total < target_int:
        return False
    
    n = len(elements_int)
    if n <= 1:
        return elements_int and elements_int[0] == target_int
    
    # Meet in the middle
    half = n // 2
    left = elements_int[:half]
    right = elements_int[half:]
    
    # Generate all subset sums for the left half
    left_sums = {0}
    for x in left:
        new_sums = set()
        for s in left_sums:
            ns = s + x
            if ns <= target_int:
                new_sums.add(ns)
        left_sums.update(new_sums)
    
    # Check if any combination of left and right sums equals target
    for rs in right:
        # For each element in right, check if target - rs can be formed by a subset of remaining right elements
        # Actually, let's generate all subset sums for right half too
        pass
    
    # Better: generate all subset sums for right half
    right_sums = {0}
    for x in right:
        new_sums = set()
        for s in right_sums:
            ns = s + x
            if ns <= target_int:
                new_sums.add(ns)
        right_sums.update(new_sums)
    
    # Check if there exist l in left_sums and r in right_sums such that l + r == target_int
    # and l != 0 or r != 0 (non-empty subset)
    if target_int in left_sums and (target_int != 0 or right_sums):
        # target_int can be formed by left alone (if non-empty)
        if target_int != 0:
            return True
    
    if target_int in right_sums:
        if target_int != 0:
            return True
    
    # Check combinations
    for l in left_sums:
        if l == 0:
            continue
        r_needed = target_int - l
        if r_needed in right_sums:
            return True
    
    return False

def _init():
    global _DEADLINE, _BEST
    _DEADLINE = time.time() + 50.0
    _BEST = None

def _subset_sum_exists(target_frac, elements):
    """
    Check if target_frac can be formed by a non-empty subset of reciprocals of elements.
    """
    if not elements:
        return False
        
    x = target_frac.denominator
    l = x
    for e in elements:
        l = _lcm(l, e)
        
    target_int = l // x
    elements_int = [l // e for e in elements]
    
    return _subset_sum_exists_int(target_int, elements_int)

def _greedy_decreasing():
    """
    Build set by adding elements from 24 down to 1.
    """
    selected = []
    for x in range(24, 0, -1):
        if _subset_sum_exists(Fraction(1, x), selected):
            continue
        else:
            selected.append(x)
    return selected

def _local_search_1_out_k_in(S):
    """
    Try to improve S by removing one element and adding multiple candidates.
    """
    S = list(S)
    improved = True
    while improved and time.time() < _DEADLINE:
        improved = False
        S_set = set(S)
        B = [x for x in range(1, 25) if x not in S_set]
        
        # Try removing each y in S
        for y in S:
            S_temp = [x for x in S if x != y]
            S_temp_set = set(S_temp)
            
            candidates = []
            for x in B:
                # Check if x is valid with respect to S_temp
                if not _subset_sum_exists(Fraction(1, x), S_temp):
                    candidates.append(x)
                    
            if len(candidates) > 1:
                # We can add all candidates
                new_S = S_temp + candidates
                S = new_S
                improved = True
                break # Restart the while loop with the new S
        
    return S

def _local_search_swap(S):
    """
    Try to improve S by swapping one element for another.
    """
    S = list(S)
    improved = True
    while improved and time.time() < _DEADLINE:
        improved = False
        S_set = set(S)
        B = [x for x in range(1, 25) if x not in S_set]
        
        for y in S:
            S_temp = [x for x in S if x != y]
            S_temp_set = set(S_temp)
            
            # Try adding each candidate from B
            for x in B:
                # Check if x is valid with respect to S_temp
                if not _subset_sum_exists(Fraction(1, x), S_temp):
                    # Check if all elements in S_temp are still valid with x added
                    new_S = S_temp + [x]
                    valid = True
                    for z in new_S:
                        others = [w for w in new_S if w != z]
                        if _subset_sum_exists(Fraction(1, z), others):
                            valid = False
                            break
                    if valid:
                        S = new_S
                        improved = True
                        break
            if improved:
                break
        
    return S

def _is_valid_set(S):
    """Check if S is a valid reciprocal-subsum-free set."""
    S_list = sorted(S, reverse=True)
    for x in S_list:
        target = Fraction(1, x)
        others = [y for y in S_list if y != x]
        if _subset_sum_exists(target, others):
            return False
    return True

def construct():
    _init()
    
    # Start with decreasing greedy
    best = _greedy_decreasing()
    
    if not best:
        best = [24]
        
    # Apply local search
    best = _local_search_1_out_k_in(best)
    
    # Try swap-based local search
    best = _local_search_swap(best)
    
    # Final check and return
    if _is_valid_set(best):
        _BEST = best
    else:
        # Fallback to greedy if local search broke it
        best = _greedy_decreasing()
        if _is_valid_set(best):
            _BEST = best
        else:
            _BEST = [24]
        
    return list(_BEST)
