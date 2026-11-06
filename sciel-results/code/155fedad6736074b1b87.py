"""Program for the suite task: find max-cardinality dissociated reciprocal subset of [1,22]."""
# EVOLVE-BLOCK-START
from math import gcd
from functools import reduce

def _lcm(a, b):
    return a * b // gcd(a, b)

_L = reduce(_lcm, range(1, 23))

# Precompute integer values for each element: L // a
_VALS = {a: _L // a for a in range(1, 23)}

def _check_dissociated(subset):
    """Check if all reciprocal subset sums are distinct.
    Uses the LCM to convert to integer sums."""
    if not subset:
        return True
    vals = [_VALS[a] for a in subset]
    k = len(vals)
    # Generate all 2^k subset sums and check for duplicates
    sums = set()
    sums.add(0)
    for v in vals:
        new_sums = set()
        for s in sums:
            ns = s + v
            if ns in sums or ns in new_sums:
                return False
            new_sums.add(ns)
        sums.update(new_sums)
    return True

def _search_max_cardinality():
    """Find the largest dissociated subset of [1,22] using backtracking."""
    candidates = list(range(1, 23))
    
    # Try from largest cardinality down
    for k in range(22, 0, -1):
        result = _backtrack(candidates, k, [])
        if result is not None:
            return result
    return [1]

def _backtrack(remaining, target_size, current):
    """Backtrack to find a valid dissociated subset of target_size from remaining."""
    if target_size == 0:
        return list(current)
    if len(remaining) < target_size:
        return None
    
    for i, cand in enumerate(remaining):
        new_current = current + [cand]
        if _check_dissociated(new_current):
            result = _backtrack(remaining[i+1:], target_size - 1, new_current)
            if result is not None:
                return result
    return None

# Cache results to avoid recomputation
def construct():
    """Return a nonempty subset of [1,22] with all distinct reciprocal subset sums, maximizing cardinality."""
    result = _search_max_cardinality()
    return result
# EVOLVE-BLOCK-END
