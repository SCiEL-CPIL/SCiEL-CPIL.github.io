"""Rewritten program for the reciprocal-subsum-free set task."""
import time
from fractions import Fraction
from itertools import combinations
from math import gcd

_START = None
DEADLINE = 50.0  # Safety margin before 90s kill

def _deadline_hit():
    global _START
    if _START is None:
        return False
    return (time.time() - _START) > DEADLINE


def _check_subset_sum(target_frac, candidates):
    """
    Check if target_frac (Fraction) can be formed by a non-empty subset sum
    of reciprocals of the given candidate integers.
    Uses meet-in-the-middle with integer arithmetic for speed.
    """
    if not candidates:
        return False
    if target_frac <= 0:
        return False
    
    n = len(candidates)
    
    # Quick check: if target is larger than sum of all reciprocals, impossible
    total = sum(Fraction(1, c) for c in candidates)
    if target_frac > total:
        return False
    
    # For small n, brute force with early termination
    if n <= 10:
        # Use integer scaling
        lcm_val = 1
        for c in candidates:
            lcm_val = lcm_val * c // gcd(lcm_val, c)
        
        target_scaled = target_frac * lcm_val
        if target_scaled.denominator != 1:
            return False
        target_int = target_scaled.numerator
        scaled_recs = [lcm_val // c for c in candidates]
        
        # Brute force subset sum
        for mask in range(1, 1 << n):
            s = 0
            for i in range(n):
                if mask & (1 << i):
                    s += scaled_recs[i]
                    if s > target_int:
                        break
            else:
                if s == target_int:
                    return True
        return False
    else:
        # Meet-in-the-middle for larger n
        half = n // 2
        left = candidates[:half]
        right = candidates[half:]
        
        # Precompute all subset sums for left half
        left_sums = set()
        for mask in range(1 << len(left)):
            s = Fraction(0, 1)
            for i in range(len(left)):
                if mask & (1 << i):
                    s += Fraction(1, left[i])
            left_sums.add(s)
        
        # Check right half subsets against left sums
        for mask in range(1 << len(right)):
            s = Fraction(0, 1)
            for i in range(len(right)):
                if mask & (1 << i):
                    s += Fraction(1, right[i])
            if (target_frac - s) in left_sums:
                # Need to ensure we're using a non-empty subset overall
                # If mask==0 and target_frac in left_sums, that's fine (left non-empty)
                # If mask!=0, right is non-empty, so overall is non-empty
                # Only issue: mask==0 AND target_frac==0, but target_frac>0 guaranteed
                return True
        return False


def _is_valid_set(S):
    """
    Check if set S (list of ints) is reciprocal-subsum-free.
    For each u in S, 1/u must not be a subset sum of {1/w : w in S, w != u}.
    """
    if len(S) < 2:
        return True
    
    S_sorted = sorted(S)
    for i, u in enumerate(S_sorted):
        target = Fraction(1, u)
        # Candidates are all other elements
        candidates = [w for j, w in enumerate(S_sorted) if j != i]
        if _check_subset_sum(target, candidates):
            return False
    return True


def _greedy_construct():
    """
    Greedy construction: add elements one by one, keeping the set valid.
    Try different orderings to find a good starting point.
    """
    best_set = []
    best_size = 0
    
    # Try adding in increasing order
    S = []
    for v in range(1, 25):
        if _deadline_hit():
            break
        # Check if adding v keeps the set valid
        test_S = S + [v]
        if _is_valid_set(test_S):
            S = test_S
    
    if len(S) > best_size:
        best_size = len(S)
        best_set = S[:]
    
    # Try adding in decreasing order
    S = []
    for v in range(24, 0, -1):
        if _deadline_hit():
            break
        test_S = S + [v]
        if _is_valid_set(test_S):
            S = test_S
    
    if len(S) > best_size:
        best_size = len(S)
        best_set = S[:]
    
    # Try skipping 1 (since 1/1 = 1 is hard to avoid)
    S = []
    for v in range(2, 25):
        if _deadline_hit():
            break
        test_S = S + [v]
        if _is_valid_set(test_S):
            S = test_S
    
    if len(S) > best_size:
        best_size = len(S)
        best_set = S[:]
    
    # Try starting from larger numbers and working down
    S = []
    for v in range(24, 0, -1):
        if _deadline_hit():
            break
        test_S = S + [v]
        if _is_valid_set(test_S):
            S = test_S
    
    if len(S) > best_size:
        best_size = len(S)
        best_set = S[:]
    
    # Try a mixed order: evens first, then odds
    S = []
    for v in [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 1]:
        if _deadline_hit():
            break
        test_S = S + [v]
        if _is_valid_set(test_S):
            S = test_S
    
    if len(S) > best_size:
        best_size = len(S)
        best_set = S[:]
    
    return best_set


def _local_search(initial_set):
    """
    Local search: try to improve the set by swapping elements.
    """
    S = set(initial_set)
    best_set = list(S)
    best_size = len(S)
    
    improved = True
    while improved and not _deadline_hit():
        improved = False
        
        # Try removing one element and adding another
        for remove in list(S):
            if _deadline_hit():
                break
            for add in range(1, 25):
                if add in S:
                    continue
                new_S = (S - {remove}) | {add}
                if len(new_S) <= best_size:
                    continue
                if _is_valid_set(list(new_S)):
                    S = new_S
                    best_set = list(S)
                    best_size = len(S)
                    improved = True
                    break
            if improved:
                break
        
        # Try adding one element
        for add in range(1, 25):
            if _deadline_hit():
                break
            if add in S:
                continue
            new_S = S | {add}
            if _is_valid_set(list(new_S)):
                S = new_S
                best_set = list(S)
                best_size = len(S)
                improved = True
    
    return best_set


def _branch_and_bound():
    """
    Branch and bound search with strong pruning.
    """
    candidates = list(range(1, 25))
    best_set = []
    best_size = 0
    
    # Sort candidates in a smart order: try to include "safe" elements first
    # Elements with large values are safer (their reciprocals are small)
    # But we also want to include small elements if possible
    
    def backtrack(idx, S):
        nonlocal best_set, best_size
        
        if _deadline_hit():
            return
        
        # Pruning
        if len(S) + (len(candidates) - idx) <= best_size:
            return
        
        if idx == len(candidates):
            if len(S) > best_size:
                best_size = len(S)
                best_set = S[:]
            return
        
        v = candidates[idx]
        
        # Option 1: Include v
        test_S = S + [v]
        if _is_valid_set(test_S):
            backtrack(idx + 1, test_S)
        
        # Option 2: Exclude v
        backtrack(idx + 1, S)
    
    backtrack(0, [])
    return best_set


def _conflict_graph_approach():
    """
    Build a conflict graph where two elements conflict if their reciprocals
    create a subsum relationship. Then find maximum independent set.
    """
    # First, identify direct conflicts: 1/a = 1/b + 1/c for distinct a,b,c
    # More generally, check if any element's reciprocal can be formed by others
    
    # Build adjacency: i conflicts with j if including both creates a violation
    # This is complex, so use a simpler heuristic:
    # Two elements "conflict" if 1/max(i,j) <= 1/min(i,j) + 1/next_min(i,j)
    # Actually, let's just use the greedy approach with better ordering
    
    # Try multiple random orderings
    import random
    best_set = []
    best_size = 0
    
    for trial in range(20):
        if _deadline_hit():
            break
        
        order = list(range(1, 25))
        random.shuffle(order)
        
        S = []
        for v in order:
            if _deadline_hit():
                break
            test_S = S + [v]
            if _is_valid_set(test_S):
                S = test_S
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S[:]
    
    return best_set


def construct():
    global _START
    _START = time.time()
    
    # Start with greedy
    result = _greedy_construct()
    
    # Improve with local search
    result = _local_search(result)
    
    # Try conflict graph approach if time permits
    if not _deadline_hit():
        cg_result = _conflict_graph_approach()
        if len(cg_result) > len(result):
            result = cg_result
    
    # Try branch and bound if time permits
    if not _deadline_hit():
        bb_result = _branch_and_bound()
        if len(bb_result) > len(result):
            result = bb_result
    
    # Final local search polish
    if not _deadline_hit():
        result = _local_search(result)
    
    # Safety: ensure nonempty
    if not result:
        result = [24]
    
    return result
