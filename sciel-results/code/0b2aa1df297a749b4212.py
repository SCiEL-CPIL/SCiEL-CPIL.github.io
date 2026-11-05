"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
from fractions import Fraction
import time
import itertools
import sys
from math import gcd

# EVOLVE-BLOCK-START
_DEADLINE = None
_BEST = None

def _lcm(a, b):
    return a // gcd(a, b) * b

def _check_safe_fast(a, others):
    """Check if 1/a can be formed by a subset of reciprocals of others.
    Uses integer arithmetic via LCM to avoid Fraction overhead."""
    if not others:
        return True
    # We need: sum of some subset of {1/x for x in others} == 1/a
    # Multiply both sides by LCM(a, others[0], others[1], ...)
    # Let L = LCM of a and all elements in others
    # Then we need: sum of subset of {L/x for x in others} == L/a
    L = a
    for x in others:
        L = _lcm(L, x)
    target = L // a
    terms = [L // x for x in others]
    
    n = len(terms)
    if n == 0:
        return True
    
    # Meet-in-the-middle for n <= 20
    if n <= 20:
        half = n // 2
        left_terms = terms[:half]
        right_terms = terms[half:]
        
        # Generate all subset sums for left half
        left_sums = set()
        for mask in range(1 << len(left_terms)):
            s = 0
            for i in range(len(left_terms)):
                if mask & (1 << i):
                    s += left_terms[i]
            left_sums.add(s)
        
        # Check if any right subset sum + left subset sum == target
        for mask in range(1 << len(right_terms)):
            s = 0
            for i in range(len(right_terms)):
                if mask & (1 << i):
                    s += right_terms[i]
            if target - s in left_sums:
                return False
        return True
    else:
        # For larger n, use greedy pruning DFS
        terms_sorted = sorted(terms, reverse=True)
        suffix_sums = [0] * (n + 1)
        for i in range(n - 1, -1, -1):
            suffix_sums[i] = suffix_sums[i + 1] + terms_sorted[i]
        
        if suffix_sums[0] < target:
            return True
        
        visited = set()
        
        def dfs(i, acc):
            if acc == target:
                return False  # Found a violation
            if acc > target:
                return True   # No violation from this branch
            if i == n:
                return True
            if acc + suffix_sums[i] < target:
                return True
            
            state = (i, acc)
            if state in visited:
                return True
            visited.add(state)
            
            if not dfs(i + 1, acc + terms_sorted[i]):
                return False
            if not dfs(i + 1, acc):
                return False
            return True
        
        return dfs(0, 0)

def _is_safe_decreasing(a, larger_elements):
    """Check if 1/a can be formed by a subset of reciprocals of larger_elements."""
    if not larger_elements:
        return True
    return _check_safe_fast(a, larger_elements)

def _greedy_decreasing():
    """Build a maximal set by adding elements from 24 down to 1."""
    S = []
    for a in range(24, 0, -1):
        if _is_safe_decreasing(a, S):
            S.append(a)
    return S

def _full_validate(S):
    """Full validation: for each a in S, check no subset of S\{a} sums to 1/a."""
    for a in S:
        others = [x for x in S if x != a]
        if not others:
            continue
        if not _check_safe_fast(a, others):
            return False
    return True

def construct():
    global _DEADLINE, _BEST
    start = time.time()
    _DEADLINE = start + 85.0  # safety margin before 90s kill

    # Phase 1: Fast greedy decreasing construction
    S = _greedy_decreasing()
    _BEST = list(S)
    
    # Phase 2: Verify the greedy result is actually valid
    if not _full_validate(S):
        changed = True
        while changed:
            if time.time() > _DEADLINE:
                break
            changed = False
            for a in sorted(S):
                if time.time() > _DEADLINE:
                    break
                others = [x for x in S if x != a]
                if others:
                    if not _check_safe_fast(a, others):
                        S.remove(a)
                        changed = True
                        break
        _BEST = list(S)
    
    # Phase 3: Try to improve by adding elements
    current_size = len(S)
    in_set = set(S)
    out_set = [x for x in range(1, 25) if x not in in_set]
    
    improved = True
    while improved and time.time() < _DEADLINE:
        improved = False
        for a in out_set:
            if time.time() > _DEADLINE:
                break
            trial = list(S) + [a]
            if _full_validate(trial):
                S = trial
                current_size = len(S)
                in_set = set(S)
                out_set = [x for x in range(1, 25) if x not in in_set]
                _BEST = list(S)
                improved = True
                break
    
    # Phase 4: More aggressive search - try different greedy orders
    if time.time() < _DEADLINE - 10:
        import random
        best_found = list(S)
        best_size = len(S)
        
        # Try several random orderings
        num_trials = 0
        while time.time() < _DEADLINE - 5 and num_trials < 50:
            num_trials += 1
            order = list(range(1, 25))
            random.shuffle(order)
            
            trial_S = []
            for a in order:
                if _is_safe_decreasing(a, [x for x in trial_S if x > a]):
                    # Full check for safety
                    if _full_validate(trial_S + [a]):
                        trial_S.append(a)
            
            if len(trial_S) > best_size:
                best_size = len(trial_S)
                best_found = trial_S
                _BEST = list(trial_S)
        
        S = best_found
        _BEST = list(S)
    
    return list(_BEST)
# EVOLVE-BLOCK-END
