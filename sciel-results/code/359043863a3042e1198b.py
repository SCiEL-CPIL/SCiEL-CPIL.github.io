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

        if backtrack(0):
            return assignment
        return None

    # Search: iterate W from minimum possible, check if any L works
    # Minimum W is N-1 (since values must be distinct)
    min_W = n - 1
    max_W = min(max_val - 1, period + max_W) # Just a safe upper bound, we'll break early
    
    for W in range(min_W, max_W + 1):
        if time.time() > deadline:
            break
            
        # Check if there exists an L in [1, period] such that [L, L+W] is feasible
        # Note: L+W must be <= max_val. If L+W > max_val, we can't use that L.
        # Max L is min(period, max_val - W)
        max_L = min(period, max_val - W)
        
        for L in range(1, max_L + 1):
            if time.time() > deadline:
                break
                
            assignment = feasibility(L, L + W)
            if assignment is not None:
                # Found a solution with width W
                result = [assignment[i] for i in range(n)]
                # Validate
                assert len(set(result)) == n
                for i in range(n):
                    assert isinstance(result[i], int)
                    assert result[i] % divisors[i] == 0
                    assert 1 <= result[i] <= max_val
                return result
        
        # If no L worked for this W, try next W

    # Fallback: if we run out of time or W range, return a valid (though not optimal) solution
    # We can just return the divisors themselves, or find any valid solution
    # A simple valid solution: pick multiples far apart
    # e.g., divisors[i] * (i+1) * some_large_number
    # But they must be distinct and <= 1000000.
    # Since N=10, divisors are small, we can just use the divisors themselves if they are distinct.
    # They are: 6, 10, 14, 15, 21, 22, 26, 33, 35, 39. All distinct.
    return [divisors[i] for i in range(n)]
