import time
from functools import lru_cache

def construct():
    n = 5040
    target = 4979
    
    # 1. Get all divisors of n
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    
    # Sort divisors in descending order to prioritize larger divisors
    # This helps find solutions with fewer terms more quickly
    divisors.sort(reverse=True)
    N = len(divisors)
    
    # 2. Dynamic Programming with memoization
    # State: (index, remaining_sum) -> minimum number of divisors needed
    # We use a large number for infinity
    INF = 10**9
    
    @lru_cache(maxsize=None)
    def dp(i, s):
        """
        Returns minimum number of divisors from divisors[i:] that sum to s.
        Returns INF if not possible.
        """
        if s == 0:
            return 0
        if i == N:
            return INF
        if s < 0:
            return INF
        
        # Option 1: Don't take divisors[i]
        val1 = dp(i + 1, s)
        
        # Option 2: Take divisors[i]
        val2 = INF
        d = divisors[i]
        if s >= d:
            sub = dp(i + 1, s - d)
            if sub < INF:
                val2 = sub + 1
        
        return min(val1, val2)
    
    # Check if solution exists
    min_count = dp(0, target)
    if min_count == INF:
        # Should not happen for practical numbers, but handle gracefully
        return []
    
    # 3. Backtrack to find the divisors
    selected = []
    s = target
    for i in range(N):
        d = divisors[i]
        # Check if taking this divisor leads to the optimal solution
        if s >= d:
            sub = dp(i + 1, s - d)
            if sub < INF and sub + 1 == dp(i, s):
                selected.append(d)
                s -= d
    
    return selected
