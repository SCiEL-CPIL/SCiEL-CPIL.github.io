import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin within 60s limit
    
    n = 5040
    target = 4979
    
    # Generate all positive divisors of 5040
    divs = []
    i = 1
    while i * i <= n:
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
        i += 1
    
    # Sort in descending order to try larger divisors first
    divs.sort(reverse=True)
    m = len(divs)
    
    # DP: dp[s] = minimum number of divisors to achieve sum s
    INF = 10**9
    dp = [INF] * (target + 1)
    dp[0] = 0
    
    # parent[s] = (previous_sum, divisor_used) for reconstruction
    parent = [None] * (target + 1)
    
    # Check if we can reach target with current set of divisors
    def can_reach(t, current_dp):
        return current_dp[t] < INF
    
    # Iterate through divisors, updating DP table
    for idx, d in enumerate(divs):
        if d > target:
            continue
        
        # Time check every few divisors
        if idx % 10 == 0 and time.time() > deadline:
            break
        
        # Update from high to low to ensure each divisor used at most once
        for s in range(target, d - 1, -1):
            if dp[s - d] + 1 < dp[s]:
                dp[s] = dp[s - d] + 1
                parent[s] = (s - d, d)
        
        # Early termination: if we've found a solution, we can't do better
        # than 1 divisor, and if dp[target] is small we might be close to optimal
        # But we need to process all divisors to ensure minimal count
        # Actually, we must process all since a later (smaller) divisor might
        # combine with earlier ones to give fewer total terms.
        # So we continue unless time runs out.
    
    # Reconstruct solution
    if dp[target] == INF:
        return []
    
    current = target
    result = []
    while current > 0:
        if parent[current] is None:
            return []
        prev, d_used = parent[current]
        result.append(d_used)
        current = prev
    
    return result
