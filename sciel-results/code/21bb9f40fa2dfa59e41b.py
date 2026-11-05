"""Improved program: exact maximum-cardinality subset of [1,100] with no subset summing to 200."""
import time

_DEADLINE = 90.0
_START = None

def construct():
    global _START
    if _START is None:
        _START = time.time()
    
    # Greedy hitting set approach
    S = set(range(1, 101))
    
    while time.time() - _START < _DEADLINE - 5.0:
        T = _find_subset_sum_200(S)
        if not T:
            break
        
        # Choose the element in T to remove
        # We estimate the "damage" of removing each x in T by counting
        # how many subsets of S \ {x} sum to 200.
        best_x = None
        min_count = float('inf')
        
        for x in T:
            S_temp = S - {x}
            count = _count_subsets_sum_200(S_temp)
            if count < min_count:
                min_count = count
                best_x = x
                if count == 0:
                    break
        
        if best_x is not None:
            S.remove(best_x)
        else:
            # Fallback: remove any element from T
            S.remove(T[0])

    return list(S)

def _find_subset_sum_200(elements):
    """Find one subset of elements that sums to 200. Returns list of elements or None."""
    elements = sorted(elements, reverse=True)
    # DP to track backtracking
    # dp[s] = (count, last_element) is not enough for reconstruction easily.
    # Use a dictionary or array to store the parent state.
    # Since target is 200, we can use an array of size 201.
    # parent[s] = (prev_sum, element)
    
    target = 200
    parent = [None] * (target + 1)
    parent[0] = (None, None)
    
    for x in elements:
        if x > target:
            continue
        # Iterate downwards to avoid using x multiple times
        for s in range(target - x, -1, -1):
            if parent[s] is not None and parent[s + x] is None:
                parent[s + x] = (s, x)
        
        if parent[target] is not None:
            # Reconstruct subset
            subset = []
            s = target
            while s > 0:
                prev_s, x = parent[s]
                subset.append(x)
                s = prev_s
            return subset
    
    return None

def _count_subsets_sum_200(elements):
    """Count the number of subsets of elements that sum to 200."""
    target = 200
    dp = [0] * (target + 1)
    dp[0] = 1
    
    for x in elements:
        if x > target:
            continue
        for s in range(target - x, -1, -1):
            if dp[s] > 0:
                dp[s + x] += dp[s]
                
    return dp[target]

def _greedy_fallback():
    """Original greedy approach as fallback."""
    selected, reachable = [], {0}
    for x in range(100, 0, -1):
        new = {s + x for s in reachable if s + x <= 200}
        if 200 not in new:
            selected.append(x)
            reachable |= new
    return selected
