"""Find 10 distinct multiples of divisors [6,10,14,15,21,22,26,33,35,39]
in a minimal-width window [L, L+W], each <= 1e6."""
import time
from math import gcd
from functools import reduce

_DIVS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
_MAXV = 1000000
_DEADLINE = 80.0

def _lcm(a, b):
    return a // gcd(a, b) * b

_LCM_ALL = reduce(_lcm, _DIVS)

def _multiples_in_window(L, W, d):
    """Return list of multiples of d in [L, L+W]."""
    if L > _MAXV:
        return []
    hi = min(L + W, _MAXV)
    # first multiple >= L
    k0 = (L + d - 1) // d
    k1 = hi // d
    if k0 > k1:
        return []
    return [k * d for k in range(k0, k1 + 1)]

def _try_assign(L, W):
    """Try to assign distinct multiples of each divisor within [L, L+W].
    Returns list of assigned values or None."""
    # Precompute candidate multiples for each divisor
    candidates = []
    for d in _DIVS:
        cands = _multiples_in_window(L, W, d)
        if not cands:
            return None
        candidates.append(cands)
    
    # Backtracking: assign in order of fewest candidates first (MRV heuristic)
    order = sorted(range(len(_DIVS)), key=lambda i: len(candidates[i]))
    assigned = [None] * len(_DIVS)
    used = set()
    
    def backtrack(idx):
        if idx == len(order):
            return True
        i = order[idx]
        for val in candidates[i]:
            if val not in used:
                used.add(val)
                assigned[i] = val
                if backtrack(idx + 1):
                    return True
                used.discard(val)
                assigned[i] = None
        return False
    
    if backtrack(0):
        return [assigned[i] for i in range(len(_DIVS))]
    return None

def _has_all_multiples(L, W):
    """Quick check: does each divisor have at least one multiple in [L, L+W]?"""
    for d in _DIVS:
        k0 = (L + d - 1) // d
        k1 = (L + W) // d
        if k0 > k1:
            return False
    return True

def construct():
    t0 = time.time()
    best_reps = None
    best_W = float('inf')
    
    # Use LCM periodicity: if L works, then L + k*LCM_ALL also works
    # So we only need to check L in [1, min(LCM_ALL, _MAXV - W)]
    # LCM_ALL = 30030, which is manageable
    
    # Phase 1: Try small W with structured scan using LCM periodicity
    for W in range(9, 300):
        if time.time() - t0 > _DEADLINE:
            break
        
        # L must be in [1, _MAXV - W]
        # By LCM periodicity, we only need to check L in [1, min(LCM_ALL, _MAXV - W)]
        # But we can be smarter: for each residue class mod LCM_ALL,
        # the pattern of available multiples is the same.
        
        # For W < max(_DIVS) - 1 = 38, not all L work.
        # For W >= 38, all L have at least one multiple of each divisor.
        
        if W >= 38:
            # Every L has multiples of all divisors in [L, L+W]
            # Try L values in [1, LCM_ALL] (by periodicity)
            # But also need L + W <= _MAXV
            l_max = min(LCM_ALL, _MAXV - W)
            for L in range(1, l_max + 1):
                if time.time() - t0 > _DEADLINE:
                    break
                reps = _try_assign(L, W)
                if reps is not None:
                    best_reps = reps
                    best_W = W
                    break
            if best_reps is not None:
                break
        else:
            # W < 38: need to check which L have multiples of all divisors
            # Use LCM periodicity: check L in [1, min(LCM_ALL, _MAXV - W)]
            l_max = min(LCM_ALL, _MAXV - W)
            for L in range(1, l_max + 1):
                if time.time() - t0 > _DEADLINE:
                    break
                # Quick check: does each divisor have a multiple in [L, L+W]?
                if not _has_all_multiples(L, W):
                    continue
                reps = _try_assign(L, W)
                if reps is not None:
                    best_reps = reps
                    best_W = W
                    break
            if best_reps is not None:
                break
    
    # If not found, do a broader random search
    if best_reps is None:
        import random
        random.seed(42)
        for _ in range(100000):
            if time.time() - t0 > _DEADLINE:
                break
            W = random.randint(40, 500)
            L = random.randint(1, _MAXV - W)
            reps = _try_assign(L, W)
            if reps is not None:
                if W < best_W:
                    best_reps = reps
                    best_W = W
                    break
    
    # Fallback: return the divisors themselves
    if best_reps is None:
        best_reps = list(_DIVS)
    
    return [int(x) for x in best_reps]
