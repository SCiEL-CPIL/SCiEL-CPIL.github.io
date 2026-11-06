import time

def construct():
    n = 5040
    target = 4979
    
    start_time = time.time()
    deadline = start_time + 50.0 # 50s safety margin

    # Generate divisors of n
    divs = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
    
    # Sort divisors (optional, but good for consistency)
    divs.sort()
    
    # DP setup
    INF = float('inf')
    dp = [INF] * (target + 1)
    used = [None] * (target + 1)
    
    dp[0] = 0
    
    for d in divs:
        if time.time() > deadline:
            # If time is running out, break and return best found so far
            # Though for this size, it's unlikely to happen
            break
        # Iterate downwards to ensure each divisor is used at most once
        for s in range(target, d - 1, -1):
            if dp[s - d] + 1 < dp[s]:
                dp[s] = dp[s - d] + 1
                used[s] = d
                
    # Check if solution exists
    if dp[target] == INF:
        return []
        
    # Backtrack to find the divisors
    result = []
    s = target
    while s > 0:
        d = used[s]
        if d is None:
            break
        result.append(d)
        s -= d
        
    return result
