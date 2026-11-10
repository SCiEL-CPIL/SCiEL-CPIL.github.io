import time
import sys
from itertools import combinations

_N = 5040
_TARGET = 4979
_DEADLINE = 85.0

def _divisors_of(n):
    divs = []
    i = 1
    while i * i <= n:
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
        i += 1
    return sorted(divs)

def _greedy_solve(divisors, target):
    """Greedy: pick largest divisors first."""
    remainder = target
    chosen = []
    for d in sorted(divisors, reverse=True):
        if d <= remainder:
            chosen.append(d)
            remainder -= d
            if remainder == 0:
                break
    return chosen if sum(chosen) == target else []

def _meet_in_the_middle(divisors, target, deadline):
    """
    Meet-in-the-middle for minimum cardinality subset sum.
    Split divisors into two halves. For each half, enumerate all subsets
    and for each sum, keep the minimum cardinality (and the subset for reconstruction).
    Then combine.
    """
    n = len(divisors)
    if n > 26:
        return None
    
    half = n // 2
    left = divisors[:half]
    right = divisors[half:]
    
    # For each half, build a dict: sum -> (min_count, subset_tuple)
    def enumerate_subsets(arr):
        m = len(arr)
        best = {}  # sum -> (count, subset)
        # Iterate over all 2^m subsets
        for mask in range(1 << m):
            s = 0
            cnt = 0
            subset = []
            for i in range(m):
                if mask & (1 << i):
                    s += arr[i]
                    cnt += 1
                    subset.append(arr[i])
            if s > target:
                continue
            if s not in best or cnt < best[s][0]:
                best[s] = (cnt, tuple(subset))
        return best
    
    start = time.time()
    
    # Enumerate left half
    left_best = enumerate_subsets(left)
    
    if time.time() > deadline:
        return None
    
    # Enumerate right half
    right_best = enumerate_subsets(right)
    
    if time.time() > deadline:
        return None
    
    # Combine: for each sum_s in left_best, need target - sum_s in right_best
    best_count = float('inf')
    best_solution = None
    
    for sum_l, (cnt_l, subset_l) in left_best.items():
        rem = target - sum_l
        if rem < 0:
            continue
        if rem in right_best:
            cnt_r, subset_r = right_best[rem]
            total_cnt = cnt_l + cnt_r
            if total_cnt < best_count:
                best_count = total_cnt
                best_solution = list(subset_l) + list(subset_r)
                if time.time() > deadline:
                    break
    
    if best_solution is not None:
        return best_solution
    return None

def _bfs_min_cardinality(divisors, target, deadline):
    """
    BFS by cardinality level. Level k: all sums achievable with exactly k divisors.
    Use a set of sums at each level. Track which divisors were used via parent pointers
    is complex; instead, for small k, we can use combinations.
    """
    # This is essentially checking combinations of size 1, 2, 3, ...
    # For this problem, the answer is likely small (<= 6 or so)
    divisors_sorted = sorted(divisors, reverse=True)
    
    # Quick check: is any single divisor equal to target?
    if target in divisors:
        return [target]
    
    # Check pairs
    div_set = set(divisors)
    for d in divisors_sorted:
        if d >= target:
            continue
        rem = target - d
        if rem in div_set and rem != d:
            return [d, rem]
    
    if time.time() > deadline:
        return None
    
    # Check triples
    # Use a set for O(1) lookup
    for i, d1 in enumerate(divisors_sorted):
        if d1 >= target:
            continue
        rem1 = target - d1
        # For each d2 < d1 (to avoid duplicates and keep sorted)
        for j in range(i + 1, len(divisors_sorted)):
            d2 = divisors_sorted[j]
            if d1 + d2 >= target:
                break
            rem2 = target - d1 - d2
            if rem2 in div_set and rem2 != d1 and rem2 != d2 and rem2 < d2:
                return [d1, d2, rem2]
    
    if time.time() > deadline:
        return None
    
    # Check quadruples: d1 > d2 > d3 > d4
    # This is O(n^4) worst case but with pruning should be fast
    n = len(divisors_sorted)
    for i in range(n):
        d1 = divisors_sorted[i]
        if d1 >= target:
            continue
        rem1 = target - d1
        for j in range(i + 1, n):
            d2 = divisors_sorted[j]
            if d1 + d2 >= target:
                break
            rem2 = target - d1 - d2
            for k in range(j + 1, n):
                d3 = divisors_sorted[k]
                if d1 + d2 + d3 >= target:
                    break
                rem3 = target - d1 - d2 - d3
                if rem3 in div_set and rem3 < d3:
                    return [d1, d2, d3, rem3]
                if time.time() > deadline:
                    return None
    
    return None

def construct():
    start = time.time()
    divisors = _divisors_of(_N)
    # Filter divisors that are too large to be useful
    divisors = [d for d in divisors if d <= _TARGET]
    
    # First, try BFS by cardinality (checks k=1,2,3,4)
    # This is fast and likely finds the optimum
    result = _bfs_min_cardinality(divisors, _TARGET, _DEADLINE - 2)
    if result is not None:
        return result
    
    # If not found, try meet-in-the-middle for up to 26 divisors
    # But first, filter: only keep divisors that could be in a small solution
    # Actually, let's just try MITM on all divisors if count is reasonable
    if len(divisors) <= 26:
        result = _meet_in_the_middle(divisors, _TARGET, _DEADLINE - 1)
        if result is not None:
            return result
    
    # Fall back to DP
    INF = 10**9
    dp = [INF] * (_TARGET + 1)
    parent = [-1] * (_TARGET + 1)
    dp[0] = 0
    
    for d in divisors:
        if time.time() > _DEADLINE - 5:
            break
        for s in range(_TARGET, d - 1, -1):
            if dp[s - d] + 1 < dp[s]:
                dp[s] = dp[s - d] + 1
                parent[s] = d
    
    if dp[_TARGET] != INF:
        result = []
        s = _TARGET
        valid = True
        while s > 0:
            d = parent[s]
            if d == -1:
                valid = False
                break
            result.append(d)
            s -= d
        
        if valid and sum(result) == _TARGET and len(set(result)) == len(result):
            return result
    
    # Final fallback
    return _greedy_solve(divisors, _TARGET)
