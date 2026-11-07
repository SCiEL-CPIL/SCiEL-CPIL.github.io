import time
from math import gcd
from functools import reduce
from collections import deque

DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
N = len(DIVISORS)
MAX_VAL = 1000000
PERIOD = reduce(lambda a, b: a * b // gcd(a, b), DIVISORS)  # 60060

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

    n = N
    divisors = DIVISORS
    max_val = MAX_VAL
    period = PERIOD

    # Precompute LCM of divisors[i:] for i=1..n-1, for fast feasibility
    suffix_lcm = [0] * n
    suffix_lcm[n - 1] = divisors[n - 1]
    for i in range(n - 2, -1, -1):
        suffix_lcm[i] = suffix_lcm[i + 1] * divisors[i] // gcd(suffix_lcm[i + 1], divisors[i])

    def feasibility(l, r):
        """Check if a valid assignment exists in [l, r] using greedy + backtracking."""
        if l < 1 or r > max_val:
            return False
        W = r - l + 1
        if W < n:
            return False

        # Build adjacency
        adj = [[] for _ in range(n)]
        for i in range(n):
            d = divisors[i]
            # First j in [0, W-1] such that l+j is divisible by divisors[i]
            rem = (-l) % d
            j = rem
            if j >= W:
                return False  # No candidate for divisor i
            while j < W:
                adj[i].append(j)
                j += d

        # Sort by number of candidates (MRV heuristic)
        order = sorted(range(n), key=lambda i: len(adj[i]))

        used = [False] * W
        assignment = [0] * n

        def backtrack(idx):
            if idx == n:
                return True
            i = order[idx]
            for j in adj[i]:
                if not used[j]:
                    used[j] = True
                    assignment[i] = l + j
                    if backtrack(idx + 1):
                        return True
                    used[j] = False
            return False

        return backtrack(0)

    # Search: iterate W from n-1 upward, check all L in one period
    best_diff = float('inf')
    best_L = -1
    best_R = -1

    for W in range(n - 1, min(best_diff, max_val - 1)):
        if time.time() > deadline:
            break

        # For this window width W, check all starting positions L in [1, period]
        # such that L+W <= max_val
        max_L = min(period, max_val - W)
        if max_L < 1:
            continue

        found = False
        for L in range(1, max_L + 1):
            if time.time() > deadline:
                break
            R = L + W
            if feasibility(L, R):
                best_diff = W
                best_L = L
                best_R = R
                found = True
                break

        if found:
            break  # Found minimal W, stop searching

    # If no solution found, use simple fallback
    if best_L == -1:
        # Try a larger window as fallback
        for W in range(n - 1, min(2000, max_val - 1)):
            if time.time() > deadline:
                break
            max_L = min(period, max_val - W)
            if max_L < 1:
                continue
            for L in range(1, max_L + 1):
                if time.time() > deadline:
                    break
                R = L + W
                if feasibility(L, R):
                    best_L = L
                    best_R = R
                    break
            if best_L != -1:
                break

    if best_L == -1:
        return [divisors[i] for i in range(n)]

    # Find the actual assignment using DFS with the known window
    l, r = best_L, best_R
    W = r - l + 1

    # Build adjacency
    adj = [[] for _ in range(n)]
    for i in range(n):
        d = divisors[i]
        rem = (-l) % d
        j = rem
        if j >= W:
            return [divisors[i] for i in range(n)]  # Fallback
        while j < W:
            adj[i].append(j)
            j += d

    # Use backtracking with MRV (most constrained first)
    order = sorted(range(n), key=lambda i: len(adj[i]))
    used = [False] * W
    assignment = [0] * n

    def backtrack(idx):
        if idx == n:
            return True
        i = order[idx]
        for j in adj[i]:
            if not used[j]:
                used[j] = True
                assignment[i] = l + j
                if backtrack(idx + 1):
                    return True
                used[j] = False
        return False

    if backtrack(0):
        result = [assignment[i] for i in range(n)]
        # Validate
        assert len(set(result)) == n
        for i in range(n):
            assert isinstance(result[i], int)
            assert result[i] % divisors[i] == 0
            assert 1 <= result[i] <= max_val
        return result
    else:
        # Fallback
        return [divisors[i] for i in range(n)]
