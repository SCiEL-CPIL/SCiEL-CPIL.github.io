"""Find the largest reciprocal-subsum-free subset of [1,24]."""
import time
import random
from fractions import Fraction
from itertools import combinations

_START = None
DEADLINE = 85.0  # safety margin under 90s

def _deadline_hit():
    global _START
    if _START is None:
        return False
    return (time.time() - _START) > DEADLINE


def _is_valid_exact(S):
    """
    Exact check: for each s in S, verify that 1/s is NOT a subset sum of {1/t : t in S\{s}}.
    Uses brute force subset enumeration for each s.
    """
    S_list = list(S)
    n = len(S_list)
    for i, s in enumerate(S_list):
        target = Fraction(1, s)
        others = [Fraction(1, t) for t in S_list if t != s]
        m = len(others)
        # Check all subsets of 'others'
        # For m up to 23, 2^23 = 8M which is too slow.
        # Use meet-in-the-middle for m > 20, else brute force.
        if m <= 20:
            # Brute force
            for mask in range(1, 1 << m):
                total = Fraction(0)
                for j in range(m):
                    if mask & (1 << j):
                        total += others[j]
                        if total > target:
                            break
                if total == target:
                    return False
        else:
            # Meet-in-the-middle
            mid = m // 2
            left = others[:mid]
            right = others[mid:]
            left_sums = {Fraction(0)}
            for x in left:
                new_sums = set()
                for val in left_sums:
                    nv = val + x
                    if nv <= target:
                        new_sums.add(nv)
                left_sums = left_sums | new_sums
            right_sums = {Fraction(0)}
            for x in right:
                new_sums = set()
                for val in right_sums:
                    nv = val + x
                    if nv <= target:
                        new_sums.add(nv)
                right_sums = right_sums | new_sums
            # Check if any left_sum + right_sum == target
            right_set = set(right_sums)
            for ls in left_sums:
                if (target - ls) in right_set:
                    return False
    return True


def _find_violation_fast(S):
    """
    Fast (approximate) violation check using Fraction DP with pruning.
    Returns (s, True) if a violation is likely, else (None, False).
    May have false negatives but no false positives (if we're careful).
    Actually, the DP approach is exact if we don't prune incorrectly.
    """
    S_list = list(S)
    for s in S_list:
        target = Fraction(1, s)
        others = [Fraction(1, t) for t in S_list if t != s]
        # Build achievable sums
        achievable = {Fraction(0)}
        for recip in others:
            new_achievable = set()
            for val in achievable:
                nv = val + recip
                if nv <= target:
                    new_achievable.add(nv)
            achievable = achievable | new_achievable
            if target in achievable:
                return (s, True)
    return (None, False)


def _greedy_remove(S, rng, use_fast=False):
    """
    Greedily remove elements until S is valid.
    """
    S = list(S)
    while True:
        if _deadline_hit():
            break
        if use_fast:
            s, has_viol = _find_violation_fast(S)
        else:
            # Use exact check but it's slow for large sets
            # For now, use fast
            s, has_viol = _find_violation_fast(S)
        if s is None:
            break
        # Remove s
        S.remove(s)
    return S


def construct():
    global _START
    _START = time.time()
    
    best = None
    best_size = 0
    
    # Strategy 1: Greedy remove from full set
    full = list(range(1, 25))
    rng = random.Random(42)
    result = _greedy_remove(full, rng)
    if _is_valid_exact(result):
        if len(result) > best_size:
            best_size = len(result)
            best = result
    
    # Strategy 2: Multiple random greedy removals with different heuristics
    for i in range(300):
        if _deadline_hit():
            break
        S = list(full)
        rng.shuffle(S)
        # Process in shuffled order: for each element, check if adding it causes a violation
        # Actually, let's do: start empty, add elements in random order if valid
        S_new = []
        for s in S:
            if _deadline_hit():
                break
            S_test = S_new + [s]
            if _find_violation_fast(S_test)[1] == False:
                S_new.append(s)
        # Verify exactly
        if _is_valid_exact(S_new):
            if len(S_new) > best_size:
                best_size = len(S_new)
                best = S_new
    
    # Strategy 3: Start from a good base and try to add
    # Base: elements 13-24
    base = list(range(13, 25))
    for i in range(300):
        if _deadline_hit():
            break
        S = list(base)
        small = list(range(1, 13))
        rng.shuffle(small)
        for s in small:
            if _deadline_hit():
                break
            S_test = S + [s]
            if _find_violation_fast(S_test)[1] == False:
                S = S_test
        if _is_valid_exact(S):
            if len(S) > best_size:
                best_size = len(S)
                best = S
    
    # Strategy 4: Local search from best
    if best is not None:
        for i in range(1000):
            if _deadline_hit():
                break
            S = list(best)
            in_S = set(S)
            not_in_S = [x for x in range(1, 25) if x not in in_S]
            if not not_in_S:
                break
            # Try swapping
            to_remove = rng.choice(S)
            to_add = rng.choice(not_in_S)
            S_test = [x for x in S if x != to_remove] + [to_add]
            if _find_violation_fast(S_test)[1] == False:
                if _is_valid_exact(S_test):
                    best = S_test
    
    # Final verification
    if best is None or not _is_valid_exact(best):
        best = list(range(13, 25))
    
    return list(best)


if __name__ == "__main__":
    result = construct()
    print(f"Result: {result}, size: {len(result)}")
    print(f"Valid: {_is_valid_exact(result)}")

