"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from itertools import combinations

_start_time = time.time()
_DEADLINE = 82.0  # safety margin before 90s kill

def _time_remaining():
    return _DEADLINE - (time.time() - _start_time)

def _subset_sums(elements):
    """Compute all subset sums of reciprocals of elements using meet-in-the-middle."""
    if not elements:
        return {Fraction(0)}
    n = len(elements)
    mid = n // 2
    left = elements[:mid]
    right = elements[mid:]
    
    left_sums = {Fraction(0)}
    for x in left:
        new_sums = set()
        for s in left_sums:
            new_sums.add(s + Fraction(1, x))
        left_sums = left_sums | new_sums
    
    right_sums = {Fraction(0)}
    for x in right:
        new_sums = set()
        for s in right_sums:
            new_sums.add(s + Fraction(1, x))
        right_sums = right_sums | new_sums
    
    all_sums = set()
    for ls in left_sums:
        for rs in right_sums:
            all_sums.add(ls + rs)
    
    return all_sums

def _is_valid(subset):
    """Check if subset is reciprocal-subsum-free."""
    n = len(subset)
    if n <= 1:
        return True
    
    for i, a in enumerate(subset):
        target = Fraction(1, a)
        others = [subset[j] for j in range(n) if j != i]
        
        if not others:
            continue
        
        # Check if target can be formed as sum of subset of reciprocals of others
        # Use meet-in-the-middle for efficiency
        m = len(others)
        mid = m // 2
        left = others[:mid]
        right = others[mid:]
        
        left_sums = {Fraction(0)}
        for x in left:
            new_sums = set()
            for s in left_sums:
                new_sums.add(s + Fraction(1, x))
            left_sums = left_sums | new_sums
        
        # For each right subset sum, check if target - rs is in left_sums
        right_sums = {Fraction(0)}
        for x in right:
            new_sums = set()
            for s in right_sums:
                new_sums.add(s + Fraction(1, x))
            right_sums = right_sums | new_sums
        
        for rs in right_sums:
            if (target - rs) in left_sums:
                # Ensure we're using a non-empty subset
                if rs != Fraction(0) or (target - rs) != Fraction(0):
                    return False
    
    return True

def _greedy_build():
    """Greedy approach: start with larger elements, add smaller ones if valid."""
    # Start with all elements 13-24 (known to be valid from seed)
    best = list(range(13, 25))
    
    # Try adding smaller elements one by one
    for cand in range(1, 13):
        if _time_remaining() < 5:
            break
        test_set = best + [cand]
        if _is_valid(test_set):
            best = test_set
    
    return best

def _try_full_and_removals():
    """Try full set, then remove elements to find valid large sets."""
    full_set = list(range(1, 25))
    
    # Try full set
    if _is_valid(full_set):
        return full_set
    
    # Try removing 1 element (23-element sets)
    for i in range(24):
        if _time_remaining() < 10:
            break
        test = [full_set[j] for j in range(24) if j != i]
        if _is_valid(test):
            return test
    
    # Try removing 2 elements (22-element sets)
    if _time_remaining() > 15:
        for i in range(24):
            for j in range(i + 1, 24):
                if _time_remaining() < 5:
                    break
                test = [full_set[k] for k in range(24) if k != i and k != j]
                if _is_valid(test):
                    return test
            if _time_remaining() < 5:
                break
    
    # Try removing 3 elements (21-element sets)
    if _time_remaining() > 20:
        for combo in combinations(range(24), 3):
            if _time_remaining() < 5:
                break
            remove_set = set(combo)
            test = [full_set[k] for k in range(24) if k not in remove_set]
            if _is_valid(test):
                return test
    
    return None

def construct():
    # Strategy 1: Try to find very large valid sets via removal
    result = _try_full_and_removals()
    if result is not None:
        return result
    
    # Strategy 2: Greedy build from known valid base
    result = _greedy_build()
    if result:
        return result
    
    # Fallback: return the seed answer
    return list(range(13, 25))
# EVOLVE-BLOCK-END
