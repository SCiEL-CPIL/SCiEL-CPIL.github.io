"""Maximum-cardinality subset of [1,100] with no subset summing to 200."""
import time

_DEADLINE = 85.0
_START = None

def construct():
    global _START
    if _START is None:
        _START = time.time()
    
    # Start with {67,...,100} = 34 elements (known safe: max pair sum 99+100=199<200,
    # min triple sum 67+68+69=204>200)
    S = set(range(67, 101))
    
    # Try to add elements from 1 to 66 in various orders
    # Order: try adding elements that are "least conflicting"
    # An element x conflicts if there's a subset of S summing to 200-x
    candidates = list(range(1, 67))
    
    # Sort by how "safe" they are - elements where 200-x is harder to form
    # Actually, just try all and keep the best
    for x in candidates:
        if time.time() - _START > _DEADLINE - 3.0:
            break
        if not _has_subset_sum(S, 200 - x):
            S.add(x)
    
    # Local search: try to add more elements via swaps
    improved = True
    while improved and time.time() - _START < _DEADLINE - 3.0:
        improved = False
        S_list = sorted(S)
        for i in range(len(S_list)):
            if time.time() - _START > _DEADLINE - 3.0:
                break
            for j in range(1, 101):
                if j in S:
                    continue
                if time.time() - _START > _DEADLINE - 3.0:
                    break
                # Try swapping S_list[i] with j
                new_S = S - {S_list[i]} | {j}
                if len(new_S) > len(S) and not _has_subset_sum_200(new_S):
                    S = new_S
                    improved = True
                    break
            if improved:
                break
        # Try pure additions
        for j in range(1, 101):
            if j in S:
                continue
            if time.time() - _START > _DEADLINE - 3.0:
                break
            if not _has_subset_sum_200(S | {j}):
                S.add(j)
                improved = True
    
    return list(S)

def _has_subset_sum(elements, target):
    """Check if any subset of elements sums to target."""
    if target < 0:
        return False
    if target == 0:
        return True
    dp = [False] * (target + 1)
    dp[0] = True
    for x in elements:
        if x > target:
            continue
        for s in range(target - x, -1, -1):
            if dp[s]:
                dp[s + x] = True
                if dp[target]:
                    return True
    return dp[target]

def _has_subset_sum_200(elements):
    """Check if any subset of elements sums to 200."""
    return _has_subset_sum(elements, 200)
