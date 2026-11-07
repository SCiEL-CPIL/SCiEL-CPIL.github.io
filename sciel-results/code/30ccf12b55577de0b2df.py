"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from itertools import product

_DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
_LCM = 30030
_MAX_VAL = 1000000
_N = 10

def _get_multiples_in_window(d, offset, R):
    """Get positions p in [0, LCM) that are multiples of d and in window [offset, offset+R] mod LCM."""
    positions = []
    for p in range(0, _LCM, d):
        if (p - offset) % _LCM <= R:
            positions.append(p)
    return positions

def _check_window_offset(offset, R):
    """Check if window [offset, offset+R] (mod LCM) has distinct multiples for all divisors.
    Returns assignment (list of positions) if feasible, else None."""
    avail = []
    for i, d in enumerate(_DIVISORS):
        positions = _get_multiples_in_window(d, offset, R)
        avail.append(positions)
    
    # Backtracking: assign distinct positions to each divisor
    used = set()
    assignment = [None] * _N
    
    def backtrack(idx):
        if idx == _N:
            return True
        for p in avail[idx]:
            if p not in used:
                used.add(p)
                assignment[idx] = p
                if backtrack(idx + 1):
                    return True
                used.remove(p)
                assignment[idx] = None
        return False
    
    if backtrack(0):
        return assignment
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    best_R = None
    best_assignment = None
    
    # Binary search on R (window size)
    lo = 1
    hi = _LCM
    
    def is_feasible(R):
        if time.time() > deadline:
            return None
        
        for offset in range(_LCM):
            if time.time() > deadline:
                return None
            result = _check_window_offset(offset, R)
            if result is not None:
                return (offset, result)
        return None
    
    # Binary search for minimum R
    while lo < hi:
        if time.time() > deadline:
            break
        mid = (lo + hi) // 2
        res = is_feasible(mid)
        if res is None:
            if time.time() > deadline:
                break
            lo = mid + 1
        else:
            offset, assignment = res
            best_R = mid
            best_assignment = assignment
            hi = mid
    
    if best_R is None:
        # Fallback: just return the divisors themselves (valid but not optimal)
        return list(_DIVISORS)
    
    # Now construct actual integers from the assignment
    min_p = min(best_assignment)
    max_p = max(best_assignment)
    S_min = max(0, 1 - min_p)
    S_max = 1000000 - max_p
    
    S = None
    if S_min <= S_max:
        S = ((S_min + _LCM - 1) // _LCM) * _LCM
        if S > S_max:
            S = None
    
    if S is None:
        return list(_DIVISORS)
    
    result = [S + p for p in best_assignment]
    
    # Verify
    for i, val in enumerate(result):
        if val <= 0 or val > 1000000:
            return list(_DIVISORS)
        if val % _DIVISORS[i] != 0:
            return list(_DIVISORS)
    if len(set(result)) != _N:
        return list(_DIVISORS)
    
    return result

# EVOLVE-BLOCK-END
