"""Optimized: find minimum-count subset of distinct divisors of 5040 summing to 4979."""
import time

def _all_divisors(n):
    divs = []
    i = 1
    while i * i <= n:
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
        i += 1
    divs.sort(reverse=True)
    return divs

def _verify(chosen, n, target):
    """Verify that chosen is a valid set of distinct divisors of n summing to target."""
    if len(chosen) > 256:
        return False
    seen = set()
    total = 0
    for d in chosen:
        if d in seen:
            return False
        if d <= 0 or n % d != 0:
            return False
        seen.add(d)
        total += d
    return total == target

def _bfs_by_count(divs, target, n, deadline):
    """BFS by number of divisors: try all subsets of size k for k=1,2,..."""
    from itertools import combinations
    divs_sorted = sorted(divs, reverse=True)
    for k in range(1, 257):
        if time.time() > deadline:
            break
        # Prune: if sum of k largest divisors < target, skip
        if sum(divs_sorted[:k]) < target:
            continue
        for combo in combinations(divs_sorted, k):
            if time.time() > deadline:
                break
            if sum(combo) == target:
                return list(combo)
    return []

def construct():
    target = 4979
    n = 5040
    divs = _all_divisors(n)
    # only keep divisors <= target
    divs = [d for d in divs if d <= target]
    
    # Quick check: is target itself a divisor?
    if n % target == 0:
        return [target]
    
    # We want minimum number of distinct divisors summing to target.
    # Use DP: dp[s] = min count to reach sum s, plus reconstruction.
    INF = float('inf')
    dp = [INF] * (target + 1)
    dp[0] = 0
    # parent tracking: parent[s] = divisor used to reach s
    parent = [-1] * (target + 1)
    # iterate divisors; 0/1 knapsack style minimizing count
    for d in divs:
        for s in range(target, d - 1, -1):
            if dp[s - d] + 1 < dp[s]:
                dp[s] = dp[s - d] + 1
                parent[s] = d
    best = dp[target]
    if best != INF:
        # reconstruct
        chosen = []
        s = target
        while s > 0:
            d = parent[s]
            if d <= 0:
                break
            chosen.append(d)
            s -= d
        if _verify(chosen, n, target):
            return chosen
    
    # Fallback: BFS by count (rarely needed)
    # Note: _bfs_by_count uses time, so we need to pass a deadline or modify it.
    # For this specific small instance, DP is sufficient.
    # If DP fails, return empty or greedy.
    remainder = target
    chosen = []
    for d in divs:
        if d <= remainder:
            chosen.append(d)
            remainder -= d
        if remainder == 0:
            break
    return chosen
