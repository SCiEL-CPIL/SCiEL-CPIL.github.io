import time

def construct():
    n = 5040
    target = 4979
    
    # Generate divisors
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    
    divisors.sort()
    N = len(divisors)
    
    # DP table: dp[i][s] = min count of divisors from first i divisors to sum to s
    # INF = large number
    INF = float('inf')
    dp = [[INF] * (target + 1) for _ in range(N + 1)]
    dp[0][0] = 0
    
    # Fill DP table
    for i in range(1, N + 1):
        d = divisors[i-1]
        for s in range(target + 1):
            # Option 1: Don't use d
            best = dp[i-1][s]
            # Option 2: Use d
            if s >= d:
                if dp[i-1][s-d] + 1 < best:
                    best = dp[i-1][s-d] + 1
            dp[i][s] = best
            
    # Check if solution exists
    if dp[N][target] == INF:
        return [] # Should not happen for practical numbers with target <= n
        
    # Backtrack to find the divisors
    result = []
    s = target
    for i in range(N, 0, -1):
        d = divisors[i-1]
        if dp[i][s] == dp[i-1][s]:
            # Did not use d
            continue
        else:
            # Used d
            result.append(d)
            s -= d
            if s == 0:
                break
                
    return result
