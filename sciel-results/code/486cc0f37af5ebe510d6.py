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
    
    # Sort divisors, order doesn't strictly matter for correctness of DP, 
    # but might affect which specific set is found if ties exist.
    # Sorting descending might find "large" divisors earlier? 
    # In 0/1 knapsack with min count, the order of items doesn't change the optimal count, 
    # just which specific items are picked in case of ties.
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
        for s in range(target, d - 1, -1):
            if dp[s - d] != INF:
                # We can form sum s-d with dp[s-d] items
                # Adding divisor i gives sum s with dp[s-d] + 1 items
                new_count = dp[s - d] + 1
                if new_count < dp[s]:
                    dp[s] = new_count
                    mask[s] = mask[s - d] | (1 << i)
    
    # Check if target is reachable
    if dp[target] == INF:
        # Should not happen for practical numbers, but handle gracefully
        return []
        
    # Reconstruct the set of divisors
    result_mask = mask[target]
    selected_divisors = []
    for i in range(N):
        if result_mask & (1 << i):
            selected_divisors.append(divisors[i])
            
    return selected_divisors
