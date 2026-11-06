"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from itertools import combinations

_START_TIME = time.time()
_DEADLINE = 85.0  # seconds, safety margin before 90s kill

def _check_conflict(S, s):
    """Check if 1/s equals sum of reciprocals of a subset of S\{s}."""
    others = [t for t in S if t != s]
    target = Fraction(1, s)
    n = len(others)
    # Check all subsets of others (but this is expensive; use pruning)
    # Sum of all others' reciprocals
    total = sum(Fraction(1, t) for t in others)
    if total < target:
        return False
    # Try subsets of increasing size
    # For efficiency, only check subsets where the sum could equal target
    # Since all reciprocals are positive, if a subset's sum exceeds target, skip
    # Use recursive backtracking with pruning
    others_sorted = sorted(others, key=lambda x: -Fraction(1, x))  # largest reciprocals first
    found = False
    def backtrack(idx, current_sum, current_set):
        nonlocal found
        if found:
            return
        if current_sum == target:
            if len(current_set) > 0:
                found = True
            return
        if idx >= n:
            return
        if current_sum > target:
            return
        # Pruning: even adding all remaining can't reach target
        remaining_sum = sum(Fraction(1, others_sorted[i]) for i in range(idx, n))
        if current_sum + remaining_sum < target:
            return
        # Include others_sorted[idx]
        backtrack(idx + 1, current_sum + Fraction(1, others_sorted[idx]), current_set + [others_sorted[idx]])
        # Exclude
        backtrack(idx + 1, current_sum, current_set)
    backtrack(0, Fraction(0), [])
    return found

def _is_valid(S):
    """Check if S is reciprocal-subsum-free."""
    S_set = set(S)
    for s in S_set:
        if _check_conflict(S_set, s):
            return False
    return True

def construct():
    global _START_TIME
    if time.time() - _START_TIME > _DEADLINE:
        return list(range(13, 25))  # fallback
    
    # Start with all elements and try to find max valid subset
    # Strategy: try to keep as many as possible
    # Use a greedy removal approach with backtracking
    
    all_elements = list(range(1, 25))
    
    # First, try the large-number heuristic (13-24) as baseline
    baseline = list(range(13, 25))
    if not _is_valid(set(baseline)):
        # Need to fix baseline
        baseline = list(baseline)
        changed = True
        while changed:
            changed = False
            for s in list(baseline):
                if _check_conflict(set(baseline), s):
                    baseline.remove(s)
                    changed = True
    
    best = list(baseline)
    
    # Try to improve by adding elements from 1-12
    candidates_to_add = list(range(1, 13))
    
    # Try adding each candidate and see if we can maintain validity
    # Use a more systematic approach: try all subsets of [1..12] to add
    # But that's 2^12 = 4096, manageable
    
    import itertools
    current_set = set(best)
    
    for size in range(1, 13):
        if time.time() - _START_TIME > _DEADLINE - 5:
            break
        found_improvement = False
        for combo in combinations(range(1, 13), size):
            if time.time() - _START_TIME > _DEADLINE - 5:
                break
            new_set = current_set | set(combo)
            if _is_valid(new_set):
                best = list(new_set)
                current_set = new_set
                found_improvement = True
                break
        if not found_improvement:
            break
    
    # Also try removing some from current best and adding different ones
    # Try each element from 1-12 individually
    for s in range(1, 13):
        if time.time() - _START_TIME > _DEADLINE - 5:
            break
        if s in current_set:
            continue
        new_set = current_set | {s}
        if _is_valid(new_set):
            best = list(new_set)
            current_set = new_set
    
    # Try to swap: remove one, add another
    for s_remove in list(current_set):
        if time.time() - _START_TIME > _DEADLINE - 5:
            break
        for s_add in range(1, 25):
            if time.time() - _START_TIME > _DEADLINE - 5:
                break
            if s_add in current_set:
                continue
            new_set = (current_set - {s_remove}) | {s_add}
            if _is_valid(new_set):
                if len(new_set) > len(best):
                    best = list(new_set)
                    current_set = new_set
    
    return best if best else list(range(13, 25))
# EVOLVE-BLOCK-END
