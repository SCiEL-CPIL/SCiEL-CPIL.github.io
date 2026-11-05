import time

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
    
    # Sort divisors. Order doesn't matter for DP correctness, but let's sort for consistency.
    divisors.sort()
    N = len(divisors)
    
    # 2. Dynamic Programming
    # dp[i][s] = minimum number of divisors from divisors[0..i-1] that sum to s
    # We use a large number for infinity
    INF = 10**9
    dp = [[INF] * (target + 1) for _ in range(N + 1)]
    take = [[False] * (target + 1) for _ in range(N + 1)]
    
    # Base case
    dp[0][0] = 0
    
    # Fill DP table
    for i in range(1, N + 1):
        d = divisors[i-1]
        for s in range(target + 1):
            # Option 1: Don't take d
            val1 = dp[i-1][s]
            # Option 2: Take d
            val2 = INF
            if s >= d:
                if dp[i-1][s-d] + 1 < val1:
                    val2 = dp[i-1][s-d] + 1
            
            if val2 < val1:
                dp[i][s] = val2
                take[i][s] = True
            else:
                dp[i][s] = val1
                take[i][s] = False
                
    # Check if solution exists
    if dp[N][target] == INF:
        # Should not happen for practical numbers, but handle gracefully
        return []

    # 3. Backtrack to find the divisors
    selected = []
    s = target
    for i in range(N, 0, -1):
        if take[i][s]:
            selected.append(divisors[i-1])
            s -= divisors[i-1]
            
    return selected
