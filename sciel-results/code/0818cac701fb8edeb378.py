import math

def construct():
    n = 5040
    target = 4979
    
    # Generate divisors
    divisors = []
    for i in range(1, int(math.isqrt(n)) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    
    divisors.sort()
    
    INF = float('inf')
    # dp[s] stores the minimum number of divisors to reach sum s
    dp = [INF] * (target + 1)
    # last_div[s] stores the divisor that was used to reach sum s optimally
    last_div = [0] * (target + 1)
    
    dp[0] = 0
    
    for d in divisors:
        # Iterate backwards to ensure each divisor is used at most once
        # We only need to check sums >= d
        for s in range(target, d - 1, -1):
            if dp[s - d] + 1 < dp[s]:
                dp[s] = dp[s - d] + 1
                last_div[s] = d
                
    if dp[target] == INF:
        return []
        
    # Backtrack
    selected = []
    s = target
    while s > 0:
        d = last_div[s]
        if d == 0:
            # This should not happen if dp[target] is not INF, 
            # but good for safety
            return []
        selected.append(d)
        s -= d
        
    return selected
