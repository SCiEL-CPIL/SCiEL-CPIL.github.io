import time

def construct():
    start_time = time.time()
    target = 4979
    n = 5040
    
    # Get divisors
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    
    # Sort in descending order to potentially find solutions faster if we were pruning, 
    # but for DP it doesn't matter much. 
    divisors.sort(reverse=True)
    
    # DP table: dp[i][j] = min number of divisors from first i divisors to sum to j
    # i ranges from 0 to len(divisors)
    # j ranges from 0 to target
    
    # Use a large number for infinity
    INF = 10**9
    
    # We can optimize space to 1D array if we don't need to backtrack easily, 
    # but backtracking is needed. 
    # Given small N (60) and T (4979), 2D list is fine.
    # 61 * 4980 integers ~ 300k.
    
    # dp[i][j]
    dp = [[INF] * (target + 1) for _ in range(len(divisors) + 1)]
    dp[0][0] = 0
    
    # choice[i][j] = True if divisor i-1 is used in optimal solution for sum j using first i divisors
    choice = [[False] * (target + 1) for _ in range(len(divisors) + 1)]
    
    for i in range(1, len(divisors) + 1):
        d = divisors[i-1]
        for j in range(target + 1):
            # Option 1: Don't use d
            dp[i][j] = dp[i-1][j]
            choice[i][j] = False
            
            # Option 2: Use d
            if j >= d:
                if dp[i-1][j-d] + 1 < dp[i][j]:
                    dp[i][j] = dp[i-1][j-d] + 1
                    choice[i][j] = True
    
    # Check if solution exists
    if dp[len(divisors)][target] == INF:
        # Fallback to greedy if DP fails (should not happen)
        remainder = target
        chosen = []
        for d in divisors:
            if d <= remainder:
                chosen.append(d)
                remainder -= d
        assert remainder == 0
        return chosen
    
    # Backtrack to find the divisors
    chosen = []
    i = len(divisors)
    j = target
    while i > 0:
        if choice[i][j]:
            chosen.append(divisors[i-1])
            j -= divisors[i-1]
        i -= 1
        
    # Verify
    assert sum(chosen) == target
    assert len(chosen) == len(set(chosen))
    for d in chosen:
        assert n % d == 0
        
    return chosen
