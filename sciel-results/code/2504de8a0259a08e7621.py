"""
Optimized search for distinct multiples with minimal spread.
Uses periodicity-aware L scanning and fast backtracking with MRV.
"""
import time
from math import gcd
from functools import reduce

# EVOLVE-BLOCK-START
_DIVS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
_MAXV = 1000000
_DEADLINE = 85.0  # wall-clock seconds (90s kill, leave margin)

def _lcm(a, b):
    return a // gcd(a, b) * b

# Precompute LCM of all divisors
_LCM_ALL = reduce(_lcm, _DIVS)

def _try_assign_fast(L, W):
    """
    Fast feasibility check: can we assign distinct multiples in [L, L+W]?
    Uses backtracking with MRV and forward checking.
    Returns list of assigned values or None.
    """
    n = len(_DIVS)
    
    # Precompute candidate multiples for each divisor
    candidates = []
    for d in _DIVS:
        k0 = (L + d - 1) // d
        k1 = (L + W) // d
        if k0 > k1:
            return None
        # Store as list of values
        cands = [k * d for k in range(k0, k1 + 1)]
        candidates.append(cands)
    
    # MRV ordering: sort by number of candidates
    order = list(range(n))
    order.sort(key=lambda i: len(candidates[i]))
    
    assigned = [None] * n
    used = set()
    
    def backtrack(depth):
        if depth == n:
            return True
        i = order[depth]
        for val in candidates[i]:
            if val in used:
                continue
            # Forward checking: verify remaining variables have candidates
            feasible = True
            for d_idx in range(depth + 1, n):
                j = order[d_idx]
                has_candidate = False
                for v in candidates[j]:
                    if v not in used and v != val:
                        has_candidate = True
                        break
                if not has_candidate:
                    feasible = False
                    break
            if not feasible:
                continue
            
            used.add(val)
            assigned[i] = val
            if backtrack(depth + 1):
                return True
            used.discard(val)
            assigned[i] = None
        return False
    
    if backtrack(0):
        return assigned
    return None

def _scan_L_for_W(W, L_start, L_end, t0):
    """
    Scan L in [L_start, L_end] for a valid assignment with window W.
    Returns (reps, L_found) or (None, None).
    """
    # Step size: we can skip L values that are "equivalent" modulo some period
    # For simplicity, scan all L but with early exit
    for L in range(L_start, L_end + 1):
        if time.time() - t0 > _DEADLINE:
            return None, None
        if L + W > _MAXV:
            break
        reps = _try_assign_fast(L, W)
        if reps is not None:
            return reps, L
    return None, None

def construct():
    t0 = time.time()
    
    # Phase 1: Fine-grained search for small W
    # W must be at least 9 (10 distinct integers)
    # Due to divisibility constraints, expect W in range [9, 50]
    
    # Strategy: For each W, scan L over one full period or a good chunk
    # The period is LCM_ALL, but that's huge. Instead, note that the pattern
    # of available multiples repeats with period lcm of any subset.
    # For practical purposes, scanning L up to ~10000 should cover most cases.
    
    # Use a smaller initial L limit for small W, expand if needed
    for W in range(9, 60):
        if time.time() - t0 > _DEADLINE - 10:
            break
        
        # For small W, the critical region is where multiples are dense
        # Scan L from 1 to a reasonable limit
        # Use L_LIMIT that scales with W but stays bounded
        L_LIMIT = min(20000, 5000 + W * 100)
        
        reps, L_found = _scan_L_for_W(W, 1, L_LIMIT, t0)
        if reps is not None:
            return [int(x) for x in reps]
        
        # If not found in first pass, try a second pass with different L range
        # Focus on regions near multiples of smaller divisors
        if time.time() - t0 > _DEADLINE - 10:
            break
        
        # Try L values that are near multiples of 6, 10, 14, 15 (the smaller divisors)
        # This is heuristic but can help find solutions faster
        tried_L = set()
        for d in [6, 10, 14, 15, 21]:
            # Multiples of d in range [1, 50000]
            k0 = 1
            k1 = 50000 // d
            for k in range(k0, k1 + 1, max(1, 50000 // (d * 20))):
                L = k * d
                if L in tried_L:
                    continue
                tried_L.add(L)
                if L + W > _MAXV:
                    continue
                reps = _try_assign_fast(L, W)
                if reps is not None:
                    return [int(x) for x in reps]
                if time.time() - t0 > _DEADLINE - 10:
                    break
            if time.time() - t0 > _DEADLINE - 10:
                break
    
    # Phase 2: Broader search for larger W if Phase 1 didn't find anything
    for W in range(60, 150):
        if time.time() - t0 > _DEADLINE - 5:
            break
        
        L_LIMIT = 10000
        reps, L_found = _scan_L_for_W(W, 1, L_LIMIT, t0)
        if reps is not None:
            return [int(x) for x in reps]
    
    # Phase 3: Random search fallback
    import random
    random.seed(42)
    best_reps = None
    best_W = float('inf')
    
    for W in range(9, 100):
        if time.time() - t0 > _DEADLINE - 2:
            break
        attempts = 2000
        for _ in range(attempts):
            if time.time() - t0 > _DEADLINE - 2:
                break
            L = random.randint(1, min(_MAXV - W, 200000))
            reps = _try_assign_fast(L, W)
            if reps is not None:
                if W < best_W:
                    best_reps = reps
                    best_W = W
    
    if best_reps is not None:
        return [int(x) for x in best_reps]
    
    # Ultimate fallback: return the divisors themselves
    return [int(d) for d in _DIVS]
# EVOLVE-BLOCK-END
