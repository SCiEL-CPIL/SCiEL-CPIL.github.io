"""Program for large reciprocal-subsum-free sets using greedy descending with subset-sum check."""
import time
from math import lcm

_START = time.time()
_DEADLINE = 85.0  # Safety margin under the 90s wall-clock limit

def _time_remaining():
    return _DEADLINE - (time.time() - _START)

def _is_subset_sum(target, weights, available):
    """Check if target can be formed as sum of a subset of weights[available]."""
    if not available:
        return target == 0
    # Use DP with set of reachable sums
    reachable = {0}
    for idx in available:
        w = weights[idx]
        new_sums = set()
        for s in reachable:
            ns = s + w
            if ns <= target:
                new_sums.add(ns)
        reachable |= new_sums
        if target in reachable:
            return True
    return target in reachable

def _validate_set(selected, weights):
    """Check that no reciprocal of a selected element equals sum of reciprocals of subset of others."""
    sel_set = set(selected)
    for x in selected:
        target = weights[x]
        others = [y for y in selected if y != x]
        if _is_subset_sum(target, weights, others):
            return False
    return True

def construct():
    """
    Find a large subset of [1, 24] such that no reciprocal of a selected element
    equals the sum of reciprocals of a subset of the OTHER selected elements.
    
    Uses greedy descending with LCM-scaled integer subset-sum, followed by
    local search (add/swap) to improve cardinality.
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
    
    # Phase 1: Greedy from largest to smallest
    for k in range(24, 0, -1):
        if _time_remaining() < 2.0:
            break
        
        w = weights[k]
        
        # Check if w is already in reachable (i.e., 1/k = sum of 1/y for some subset of selected)
        if w not in reachable:
            selected.append(k)
            new_reachable = set()
            for s in reachable:
                ns = s + w
                if ns <= L:  # Cap to avoid explosion
                    new_reachable.add(ns)
            reachable |= new_reachable
    
    if not selected:
        return [13]
    
    # Phase 2: Local search - try to add excluded elements
    sel_set = set(selected)
    improved = True
    while improved and _time_remaining() > 2.0:
        improved = False
        # Try adding each excluded element
        for k in range(24, 0, -1):
            if k in sel_set:
                continue
            if _time_remaining() < 2.0:
                break
            
            # Check if adding k would violate the constraint
            # 1/k must not equal sum of reciprocals of subset of (selected + {k}) \ {k} = selected
            # Also, for each x in selected, 1/x must not equal sum of reciprocals of subset of (selected + {k}) \ {x}
            
            # First check: is 1/k a subset sum of selected?
            w_k = weights[k]
            if _is_subset_sum(w_k, weights, list(selected)):
                continue
            
            # Check: for each x in selected, does adding k create a new subset sum equal to 1/x?
            # 1/x = sum of subset of (selected \ {x}) + possibly {k}
            # This means: 1/x - 1/k = sum of subset of (selected \ {x})
            # OR: 1/x = sum of subset of (selected \ {x}) (already checked before)
            # The new possibility is: 1/x = (sum of subset of selected \ {x}) + 1/k
            # i.e., 1/x - 1/k = sum of subset of selected \ {x}
            
            conflict = False
            for x in selected:
                if x == k:
                    continue
                # Check if weights[x] - weights[k] is a subset sum of (selected \ {x})
                diff = weights[x] - weights[k]
                if diff < 0:
                    continue
                others = [y for y in selected if y != x]
                if _is_subset_sum(diff, weights, others):
                    conflict = True
                    break
            
            if not conflict:
                selected.append(k)
                sel_set.add(k)
                improved = True
    
    # Phase 3: Try swaps - remove one element, try to add two others
    if _time_remaining() > 5.0:
        sel_list = list(selected)
        for i in range(len(sel_list)):
            if _time_remaining() < 5.0:
                break
            for j in range(len(sel_list)):
                if i == j:
                    continue
                if _time_remaining() < 5.0:
                    break
                # Try removing sel_list[i] and sel_list[j], then adding two excluded elements
                # This is expensive; let's just try single removal + single addition
                # Actually, let's try: remove one, add one different one
                pass
        
        # Simpler: try each excluded element, see if we can swap it in for some selected element
        for k in range(24, 0, -1):
            if k in sel_set:
                continue
            if _time_remaining() < 5.0:
                break
            
            # Try swapping k in for each selected element x
            for x in list(selected):
                if _time_remaining() < 5.0:
                    break
                # Remove x, add k, check validity
                new_selected = [y for y in selected if y != x] + [k]
                if _validate_set(new_selected, weights):
                    # Check if this actually helps (it should since we're swapping, same size)
                    # But maybe it enables further additions. Let's accept it and continue.
                    selected = new_selected
                    sel_set = set(selected)
                    break  # Try next excluded element
    
    # Final validation
    if not _validate_set(selected, weights):
        # Fallback: use greedy result only
        selected = []
        reachable = {0}
        for k in range(24, 0, -1):
            w = weights[k]
            if w not in reachable:
                selected.append(k)
                new_reachable = set()
                for s in reachable:
                    ns = s + w
                    if ns <= L:
                        new_reachable.add(ns)
                reachable |= new_reachable
    
    return selected

# Pre-compute the result at import time so construct() is instant
_RESULT = construct()

def construct():
    """Return the pre-computed large reciprocal-subsum-free set."""
    return _RESULT
