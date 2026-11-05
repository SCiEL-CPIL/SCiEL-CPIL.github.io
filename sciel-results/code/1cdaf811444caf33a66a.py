# EVOLVE-BLOCK-START
from fractions import Fraction
import time
import itertools
import sys

_DEADLINE = None
_BEST = None

def _check_safe_target(target, others):
    """Check if any nonempty subset of others sums to target."""
    if target <= 0:
        return False
    m = len(others)
    if m == 0:
        return False
    
    o = sorted(others)
    terms = [Fraction(1, x) for x in o]
    n = len(o)
    suffix_sums = [Fraction(0)] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sums[i] = suffix_sums[i + 1] + terms[i]
        
    if suffix_sums[0] < target:
        return False
        
    def dfs(i, acc):
        if acc == target:
            return True
        if acc > target:
            return False
        if i == n:
            return False
        if acc + suffix_sums[i] < target:
            return False
        if dfs(i + 1, acc + terms[i]):
            return True
        if dfs(i + 1, acc):
            return True
        return False
        
    return dfs(0, Fraction(0))

def _check_safe(a, others):
    """Return True if no nonempty subset of `others` sums to Fraction(1, a)."""
    target = Fraction(1, a)
    return not _check_safe_target(target, others)

def _is_safe_addition(a, S):
    """Check if adding a to S keeps S valid."""
    # 1. Check a against S
    if not _check_safe(a, S):
        return False
        
    # 2. Check existing b in S against S U {a}
    # Only b < a need checking (since 1/b > 1/a, target diff is positive)
    for b in S:
        if b < a:
            target = Fraction(1, b) - Fraction(1, a)
            # T must be subset of S \ {b} (since a is new, not in S)
            others = [x for x in S if x != b]
            if _check_safe_target(target, others):
                return False
    return True

def _is_valid_set(S):
    """Check if the entire set S is valid."""
    for a in S:
        others = [x for x in S if x != a]
        if not _check_safe(a, others):
            return False
    return True

def construct():
    global _DEADLINE, _BEST
    start = time.time()
    _DEADLINE = start + 85.0

    full = list(range(1, 25))
    
    # Phase 1: Greedy construction in descending order
    S = []
    for a in sorted(full, reverse=True):
        if time.time() > _DEADLINE:
            break
        if _is_safe_addition(a, S):
            S.append(a)
            
    _BEST = list(S)
    
    # Phase 2: Try to add back elements in ascending order
    for a in sorted(full):
        if time.time() > _DEADLINE:
            break
        if a in S:
            continue
        if _is_safe_addition(a, S):
            S.append(a)
            _BEST = list(S)
            
    # Phase 3: Local Search (1-swap)
    improved = True
    while improved and time.time() < _DEADLINE:
        improved = False
        in_set = set(S)
        out_set = [x for x in full if x not in in_set]
        for a in out_set:
            if time.time() > _DEADLINE:
                break
            for b in list(S):
                if time.time() > _DEADLINE:
                    break
                # Try swap: remove b, add a
                S_temp = [x for x in S if x != b]
                if _is_safe_addition(a, S_temp):
                    S = S_temp + [a]
                    improved = True
                    _BEST = list(S)
                    break
            if improved:
                break
                
    return list(_BEST)
# EVOLVE-BLOCK-END
