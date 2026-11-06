"""Program for large reciprocal-subsum-free sets using greedy descending with subset-sum check."""
import time
from math import lcm

_START = time.time()
_DEADLINE = 85.0  # Safety margin under the 90s wall-clock limit

def _time_remaining():
    return _DEADLINE - (time.time() - _START)

def construct():
    """
    Find a large subset of [1, 24] such that no reciprocal of a selected element
    equals the sum of reciprocals of a subset of the OTHER selected elements.
    
    Key insight: If all elements in a subset T are larger than x, then
    sum(1/y for y in T) can potentially equal 1/x. But if any element in T is
    smaller than x, then 1/y > 1/x for that y, so the sum > 1/x.
    
    Therefore, when building a set greedily from largest to smallest, we only need
    to check if 1/k can be formed as a sum of reciprocals of already-selected
    (larger) elements. If not, we can safely add k, and no future (smaller) elements
    can create a conflict for k because their reciprocals are larger.
    
    We use integer arithmetic: convert 1/x to LCM/x where LCM = lcm(1,...,24).
    Then the problem becomes a subset-sum problem on the integers LCM/x.
    """
    if _time_remaining() < 1.0:
        return list(range(13, 25))  # Fallback: safe small set
    
    # Compute LCM of 1 to 24
    L = 1
    for i in range(2, 25):
        L = lcm(L, i)
    
    # Weight for each element k is L // k
    weights = {k: L // k for k in range(1, 25)}
    
    selected = []
    # reachable sums of reciprocals (as integers L//k), using selected elements
    reachable = {0}
    
    # Process from largest to smallest
    for k in range(24, 0, -1):
        if _time_remaining() < 1.0:
            break
        
        w = weights[k]
        
        # Check if w is already in reachable (i.e., 1/k = sum of 1/y for some subset of selected)
        # But we need to check subsets of OTHER elements, not including k itself.
        # Since we haven't added k yet, all elements in `selected` are larger than k,
        # and none of them is k, so this is correct.
        if w not in reachable:
            # Safe to add k
            selected.append(k)
            # Update reachable sums: for each existing sum s, we can now also reach s + w
            new_reachable = set()
            for s in reachable:
                new_reachable.add(s + w)
            reachable |= new_reachable
    
    if not selected:
        return [13]
    
    return selected

# Pre-compute the result at import time so construct() is instant
_RESULT = construct()

def construct():
    """Return the pre-computed large reciprocal-subsum-free set."""
    return _RESULT
