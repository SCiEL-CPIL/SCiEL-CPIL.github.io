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
    
    # Sort divisors, order doesn't strictly matter for correctness of min count,
    # but might affect which specific solution is found if there are ties.
    # We just need ANY minimal solution.
    divisors.sort()
    
    INF = 10**9
    
    # 1D DP for min count
    dp = [INF] * (target + 1)
    dp[0] = 0
    
    # Backtracking arrays
    # prev[j] stores the previous sum value
    # used[j] stores the divisor added to reach j from prev[j]
    prev = [-1] * (target + 1)
    used = [0] * (target + 1)
    
    for d in divisors:
        # Iterate backwards to ensure each divisor is used at most once
        for j in range(target, d - 1, -1):
            if dp[j - d] + 1 < dp[j]:
                dp[j] = dp[j - d] + 1
                prev[j] = j - d
                used[j] = d
                
        # Optional: Time check
        if time.time() - start_time > 50:
             break

    if dp[target] == INF:
        # Fallback (should not happen)
        return []

    # Backtrack
    chosen = []
    j = target
    while j > 0:
        d = used[j]
        if d == 0:
            # Should not happen if dp[target] < INF
            break
        chosen.append(d)
        j = prev[j]
        
    # Verify
    if sum(chosen) != target:
        return []
    if len(chosen) != len(set(chosen)):
        return []
    for d in chosen:
        if n % d != 0:
            return []
            
    return chosen
