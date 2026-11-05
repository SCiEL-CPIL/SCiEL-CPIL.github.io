import math
import time

def construct():
    n = 5040
    target = 4979
    deadline = time.time() + 50.0  # 50s limit to be safe

    # Generate divisors of 5040
    divisors = []
    for i in range(1, int(math.isqrt(n)) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    
    # Filter and sort
    divisors = [d for d in divisors if d <= target]
    divisors.sort()
    N = len(divisors)
    
    # Early check for K=1
    if target in divisors:
        return [target]

    # Early check for K=2
    # Use set for O(1) lookup
    d_set = set(divisors)
    for d1 in divisors:
        rem = target - d1
        if rem in d_set and rem != d1:
            return [d1, rem]

    # Early check for K=3
    # O(N^2)
    for i in range(N):
        d1 = divisors[i]
        if d1 >= target: break
        for j in range(i+1, N):
            d2 = divisors[j]
            rem = target - d1 - d2
            if rem <= 0: break
            if rem in d_set and rem != d1 and rem != d2:
                return [d1, d2, rem]

    # DP for min items (K >= 4)
    INF = 10**9
    dp = [INF] * (target + 1)
    dp[0] = 0
    parent = [0] * (target + 1)
    
    # Standard 0/1 knapsack for min items
    for d in divisors:
        if time.time() > deadline:
            break
        # Optimization: only update s where dp[s-d] is not INF
        # But simple loop is fine.
        for s in range(target, d - 1, -1):
            if dp[s - d] + 1 < dp[s]:
                dp[s] = dp[s - d] + 1
                parent[s] = d

    if dp[target] >= INF:
        return []

    # Reconstruct
    chosen = []
    s = target
    while s > 0:
        d_val = parent[s]
        if d_val == 0:
            # This should not happen if dp[target] < INF
            return []
        chosen.append(d_val)
        s -= d_val
        
    # Verification
    if sum(chosen) != target or len(set(chosen)) != len(chosen):
        return []
        
    return chosen
