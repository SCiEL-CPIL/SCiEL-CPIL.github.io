"""Program for large integer sets with unique triple sums."""
# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations_with_replacement

def _triple_sums(arr):
    """Return all unordered triple sums (with replacement) as a sorted list."""
    n = len(arr)
    if n < 3:
        return []
    sums = []
    for i in range(n):
        for j in range(i, n):
            for k in range(j, n):
                sums.append(arr[i] + arr[j] + arr[k])
    return sums

def _has_unique_triple_sums(arr):
    """Check if all unordered triple sums (with replacement) are unique."""
    n = len(arr)
    if n < 3:
        return True
    sums = _triple_sums(arr)
    return len(sums) == len(set(sums))

def _try_add(arr, candidate):
    """Try to add candidate to arr, maintaining sorted order and unique triple sums."""
    if candidate in arr:
        return None
    new_arr = sorted(arr + [candidate])
    if _has_unique_triple_sums(new_arr):
        return new_arr
    return None

def _greedy_build(max_val=4096, target_size=80):
    """Greedily build a set with unique triple sums."""
    selected = []
    # Start with small values to leave room for more elements
    for x in range(1, max_val + 1):
        if len(selected) >= target_size:
            break
        new_arr = _try_add(selected, x)
        if new_arr is not None:
            selected = new_arr
    return selected

def _local_search(arr, max_val=4096, max_iters=200000):
    """Local search: try to swap out larger elements for smaller unused ones to pack more."""
    arr = sorted(arr)
    best = list(arr)
    best_size = len(best)
    used = set(best)
    
    # Try to add elements by removing one and adding two smaller ones, or just adding
    # First, try simple additions
    for x in range(1, max_val + 1):
        if x in used:
            continue
        new_arr = _try_add(best, x)
        if new_arr is not None:
            best = new_arr
            used = set(best)
            best_size = len(best)
            if best_size >= 80:
                break
    
    # Now try swaps: remove one element, add one smaller unused element
    # to potentially allow adding more later
    for _ in range(max_iters):
        improved = False
        arr_sorted = sorted(best)
        used = set(best)
        
        # Try removing the largest element and adding a smaller unused one
        for i in range(len(arr_sorted) - 1, -1, -1):
            removed = arr_sorted[i]
            if removed <= 1:
                break
            remaining = [x for x in arr_sorted if x != removed]
            # Try to add smaller unused values to the remaining set
            for x in range(1, removed):
                if x in used:
                    continue
                new_arr = _try_add(remaining, x)
                if new_arr is not None:
                    # Check if we can add even more
                    best_candidate = new_arr
                    best_used = set(new_arr)
                    for y in range(1, max_val + 1):
                        if y in best_used:
                            continue
                        na2 = _try_add(best_candidate, y)
                        if na2 is not None:
                            best_candidate = na2
                            best_used = set(na2)
                            if len(best_candidate) > len(best):
                                break
                    if len(best_candidate) >= len(best):
                        best = best_candidate
                        improved = True
                        break
            if improved:
                break
        
        if not improved:
            break
    
    return sorted(best)

def construct():
    """Construct a set of up to 80 distinct positive integers <= 4096 with unique triple sums."""
    # Strategy: Use a base-3 inspired construction for guaranteed validity,
    # then greedily pack more elements.
    
    # Phase 1: Build a guaranteed-valid sparse set using exponential growth
    # Start with small base to allow more elements
    base = 3
    sparse = []
    val = 1
    while val <= 4096 and len(sparse) < 80:
        sparse.append(val)
        val *= base
    
    # Phase 2: Greedily add elements from 1 upward
    selected = []
    for x in range(1, 4097):
        if len(selected) >= 80:
            break
        new_arr = _try_add(selected, x)
        if new_arr is not None:
            selected = new_arr
    
    # Phase 3: Local search to improve
    if len(selected) < 80:
        improved = _local_search(selected, max_val=4096, max_iters=50000)
        if len(improved) > len(selected):
            selected = improved
    
    return selected
# EVOLVE-BLOCK-END
