"""Maximize cardinality of a dissociated reciprocal subset of [1,22].

Strategy:
  Phase 1 (fast): Greedy construction with float64 checks and local search.
                  This is much faster than backtracking for finding large sets.
  Phase 2 (exact): verify the best float64 candidates with exact Fraction
                  arithmetic to guarantee validity.
"""
import time
import random
from fractions import Fraction

_DEADLINE = 75.0  # seconds; safety margin before the 90 s evaluator kill
_START = time.time()

def _deadline_hit():
    return (time.time() - _START) > _DEADLINE

def _exact_valid(subset):
    """Return True iff all reciprocal subset sums (incl. empty) are distinct.
    Uses exact integer representation: sum of 1/a over a subset = (sum of
    product-of-others) / LCM. Compare the integer numerators."""
    n = len(subset)
    if n == 0:
        return True
    from math import gcd
    # LCM of the elements
    def _lcm(a, b):
        return a // gcd(a, b) * b
    L = 1
    for a in subset:
        L = _lcm(L, a)
        # early bail if L gets astronomically large (shouldn't for [1..22])
    # For each subset, numerator = sum over a in subset of (L // a)
    # Represent subset by bitmask; there are 2^n of them.
    m = 1 << n
    seen = set()
    # precompute L//a
    w = [L // a for a in subset]
    # DP over masks: num[mask] = sum of w[i] for i in mask
    nums = [0] * m
    for mask in range(1, m):
        # take lowest set bit
        lb = mask & (-mask)
        i = lb.bit_length() - 1
        prev = mask ^ lb
        nums[mask] = nums[prev] + w[i]
    for mask in range(m):
        v = nums[mask]
        if v in seen:
            return False
        seen.add(v)
    return True

def _fast_float_check(sums):
    """Check if a list of float sums is distinct."""
    if not sums:
        return True
    s = sorted(sums)
    for i in range(1, len(s)):
        if abs(s[i] - s[i-1]) < 1e-12:
            return False
    return True

def _get_sums(subset):
    """Compute all subset sums for a subset."""
    n = len(subset)
    if n == 0:
        return [0.0]
    recip = [1.0 / a for a in subset]
    m = 1 << n
    sums = [0.0] * m
    for mask in range(1, m):
        lb = mask & (-mask)
        i = lb.bit_length() - 1
        sums[mask] = sums[mask ^ lb] + recip[i]
    return sums

def construct():
    global _START
    _START = time.time()
    random.seed()
    full = list(range(1, 23))
    
    # 1. Greedy Construction with Random Restarts
    best_set = []
    best_k = 0
    
    # Seed with a known good set
    seed = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    if _fast_float_check(_get_sums(seed)):
        best_set = seed[:]
        best_k = len(seed)
    
    # Greedy with random restarts
    for restart in range(50):
        if _deadline_hit():
            break
        order = full[:]
        random.shuffle(order)
        current = []
        current_sums = [0.0]
        
        for a in order:
            if a in current:
                continue
            # Try adding a
            r = 1.0 / a
            new_sums = current_sums + [s + r for s in current_sums]
            if _fast_float_check(new_sums):
                current.append(a)
                current_sums = new_sums
                if len(current) > best_k:
                    best_k = len(current)
                    best_set = current[:]
        
    # 2. Local Search (Add)
    cur = best_set[:]
    improved = True
    while improved and not _deadline_hit():
        improved = False
        cur_sums = _get_sums(cur)
        for a in full:
            if a in cur:
                continue
            r = 1.0 / a
            new_sums = cur_sums + [s + r for s in cur_sums]
            if _fast_float_check(new_sums):
                cur.append(a)
                improved = True
                break
        
    if len(cur) > len(best_set):
        best_set = cur
        best_k = len(cur)

    # 3. Exact Verification
    if _exact_valid(best_set):
        return list(best_set)
    
    # Fallback
    if _exact_valid(seed):
        return list(seed)
    
    return [1]
