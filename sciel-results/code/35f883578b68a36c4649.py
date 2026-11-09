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
    
    # Sort divisors. 
    # Sorting descending allows us to potentially find solutions with larger divisors first,
    # but for DP min-count, the order of processing items doesn't change the optimal count.
    # However, it might affect which specific subset is chosen in case of ties.
    # We keep it descending as in the seed.
    divisors.sort(reverse=True)
    
    N = len(divisors)
    
    # DP arrays
    # dp[s] stores the minimum number of divisors to sum to s
    # mask[s] stores the bitmask of divisors used
    INF = float('inf')
    dp = [INF] * (target + 1)
    mask = [0] * (target + 1)
    
    dp[0] = 0
    mask[0] = 0
    
    # 0/1 Knapsack DP
    # We iterate through each divisor
    for i in range(N):
        d = divisors[i]
        # We must iterate s backwards to ensure each divisor is used at most once
        # s goes from target down to d
        # Optimization: Local variable access
        dp_local = dp
        mask_local = mask
        bit = 1 << i
        for s in range(target, d - 1, -1):
            prev = dp_local[s - d]
            if prev != INF:
                new_count = prev + 1
                if new_count < dp_local[s]:
                    dp_local[s] = new_count
                    mask_local[s] = mask_local[s - d] | bit
    
    # Check if target is reachable
    if dp[target] == INF:
        return []
        
    # Reconstruct the set of divisors
    result_mask = mask[target]
    selected_divisors = []
    for i in range(N):
        if result_mask & (1 << i):
            selected_divisors.append(divisors[i])
            
    return selected_divisors
