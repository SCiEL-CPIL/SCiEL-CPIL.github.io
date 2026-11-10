"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

P = {'divisors': [6, 10, 14, 15, 21, 22, 26, 33, 35, 39], 'maximum': 1000000}

def _find_multiple_in_range(d, lo, hi):
    """Find the smallest multiple of d in [lo, hi], or None if no such multiple exists."""
    if hi < lo:
        return None
    # Smallest multiple of d >= lo
    m = ((lo + d - 1) // d) * d
    if m > hi:
        return None
    return m

def _check_window(L, W, divisors):
    """Check if there exist distinct multiples of each divisor in [L, L+W].
    Returns the assignment if valid, else None.
    Uses greedy: for each divisor, pick the smallest available multiple in range.
    Then verify distinctness; if conflict, try next multiple for that divisor."""
    hi = L + W
    if hi > P['maximum']:
        return None
    
    # For each divisor, get list of multiples in [L, hi]
    candidates = []
    for d in divisors:
        m = _find_multiple_in_range(d, L, hi)
        if m is None:
            return None
        candidates.append((d, m))
    
    # Greedy assignment with conflict resolution
    assigned = [None] * len(divisors)
    used = set()
    
    # Sort by number of options (most constrained first)
    # But first, let's just try the smallest multiple for each
    for i in range(len(divisors)):
        d = divisors[i]
        m = _find_multiple_in_range(d, L, hi)
        if m is None:
            return None
        # Find a multiple not already used
        while m <= hi:
            if m not in used:
                break
            m += d
        if m > hi:
            return None
        assigned[i] = m
        used.add(m)
    
    # Verify all constraints
    for i in range(len(divisors)):
        if assigned[i] is None or assigned[i] % divisors[i] != 0:
            return None
        if assigned[i] < L or assigned[i] > hi:
            return None
    
    # Check distinctness
    if len(set(assigned)) != len(assigned):
        return None
    
    return assigned

def _search_window_width(W, divisors):
    """Try to find a valid assignment with window width W.
    Scan L from 1 to maximum-W, checking each window."""
    max_val = P['maximum']
    # We need L such that L+W <= max_val, so L <= max_val - W
    # Also L >= 1
    # For efficiency, we can skip L values where the window is "similar"
    # Key insight: the set of multiples available changes only when L crosses a multiple boundary
    
    # Simple scan with early termination
    for L in range(1, max_val - W + 1):
        result = _check_window(L, W, divisors)
        if result is not None:
            return result
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    divisors = P['divisors']
    max_val = P['maximum']
    
    # Binary search on window width W
    # Lower bound: 0 (impossible due to distinctness, but try small)
    # Upper bound: max_val - 1
    lo = 1
    hi = max_val - 1
    best_result = None
    best_W = hi + 1
    
    while lo <= hi and time.time() < deadline:
        mid = (lo + hi) // 2
        result = _search_window_width(mid, divisors)
        if result is not None:
            best_result = result
            best_W = mid
            hi = mid - 1
        else:
            lo = mid + 1
    
    if best_result is not None:
        return [int(x) for x in best_result]
    
    # Fallback: just return the divisors themselves (valid but suboptimal)
    return [int(d) for d in divisors]
# EVOLVE-BLOCK-END
