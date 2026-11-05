"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
from math import gcd
from functools import reduce
import time
import sys

# EVOLVE-BLOCK-START
_DEADLINE = None
_BEST = None

def _lcm(nums):
    """Compute LCM of a list of integers."""
    if not nums:
        return 1
    return reduce(lambda a, b: a * b // gcd(a, b), nums)

def _check_safe_fast(a, others):
    """
    Return True if no nonempty subset of `others` sums to 1/a.
    Uses meet-in-the-middle with integer arithmetic for speed.
    """
    n = len(others)
    if n == 0:
        return True
    
    # Scale all reciprocals by LCM of others and a
    # 1/x * LCM = LCM / x, which is an integer
    lcm_val = _lcm(others + [a])
    
    # Target: 1/a * lcm_val = lcm_val / a
    target = lcm_val // a
    
    # Terms: 1/x * lcm_val = lcm_val // x for each x in others
    terms = [lcm_val // x for x in others]
    
    # Sort terms in descending order for better pruning
    terms.sort(reverse=True)
    n = len(terms)
    
    # Meet in the middle: split into two halves
    mid = n // 2
    left_terms = terms[:mid]
    right_terms = terms[mid:]
    
    # Generate all subset sums for left half
    left_sums = set()
    for mask in range(1 << len(left_terms)):
        s = 0
        for i in range(len(left_terms)):
            if mask & (1 << i):
                s += left_terms[i]
                if s > target:
                    break
        if s <= target:
            left_sums.add(s)
    
    # Check if any subset sum from right half + any from left equals target
    # First, check if target is in left_sums (subset only from left)
    if target in left_sums:
        return False
    
    # Generate subset sums for right half and check
    for mask in range(1 << len(right_terms)):
        s = 0
        for i in range(len(right_terms)):
            if mask & (1 << i):
                s += right_terms[i]
                if s > target:
                    break
        if s > target:
            continue
        # Check if target - s is in left_sums
        if target - s in left_sums:
            # But we need at least one element total; since s > 0 or left_sums has non-empty, 
            # and we excluded mask=0 for left? Actually left_sums includes 0 (mask=0).
            # We need to ensure the combined subset is nonempty.
            # If s == 0 and (target - s) == 0, that's empty subset. But target > 0 always.
            # So if target - s in left_sums, and target > 0, then either s > 0 or (target-s) > 0.
            # The only issue is if both are 0, which can't happen since target >= 1.
            # So any match implies a nonempty subset.
            return False
    
    return True

def _violates(a, S):
    """True if a in S violates: some subset of S\{a} sums to 1/a."""
    others = [x for x in S if x != a]
    return not _check_safe_fast(a, others)

def _is_valid_set(S):
    """Check if the entire set S is valid."""
    for a in S:
        if _violates(a, S):
            return False
    return True

def construct():
    global _DEADLINE, _BEST
    start = time.time()
    _DEADLINE = start + 85.0  # safety margin before 90s kill

    full = list(range(1, 25))
    
    # Phase 1: Greedy removal of violators
    S = list(full)
    changed = True
    while changed:
        if time.time() > _DEADLINE:
            break
        changed = False
        for a in sorted(S):
            if time.time() > _DEADLINE:
                break
            if _violates(a, S):
                S.remove(a)
                changed = True
                break  # restart scan after removal
    
    _BEST = list(S)
    
    # Phase 2: Try to add back removed elements
    removed = [x for x in full if x not in S]
    for a in sorted(removed, reverse=True):
        if time.time() > _DEADLINE:
            break
        trial = S + [a]
        if _is_valid_set(trial):
            S = trial
    
    _BEST = list(S)
    
    # Phase 3: Local search (1-swap)
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
                trial = [x for x in S if x != b] + [a]
                if _is_valid_set(trial) and len(trial) > len(S):
                    S = trial
                    improved = True
                    break
            if improved:
                break
    
    _BEST = list(S)
    
    # Phase 4: Try to find larger valid sets via more aggressive search
    # Since _check_safe_fast is much faster, we can try more combinations
    if time.time() < _DEADLINE:
        current_size = len(S)
        in_set = set(S)
        out_set = [x for x in full if x not in in_set]
        
        # Try adding each out element and see if we can remove fewer than 1 to keep validity
        # Actually, let's try all pairs of out elements to add
        for i in range(len(out_set)):
            if time.time() > _DEADLINE:
                break
            for j in range(i + 1, len(out_set)):
                if time.time() > _DEADLINE:
                    break
                trial = S + [out_set[i], out_set[j]]
                if _is_valid_set(trial):
                    S = trial
                    in_set = set(S)
                    out_set = [x for x in full if x not in in_set]
                    break
        
        # Try single additions again after pair additions
        for a in out_set:
            if time.time() > _DEADLINE:
                break
            trial = S + [a]
            if _is_valid_set(trial):
                S = trial
    
    _BEST = list(S)
    
    return list(_BEST)
# EVOLVE-BLOCK-END
