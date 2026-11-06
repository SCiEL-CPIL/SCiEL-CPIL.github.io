import time

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    n = 5040
    target = 4979
    
    # Get all divisors of 5040
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    divisors.sort()
    
    # Filter divisors that are <= target, as they cannot be part of a sum equal to target
    # if they exceed it (since all divisors are positive).
    # Note: A divisor > target cannot be used.
    valid_divisors = [d for d in divisors if d <= target]
    
    # DP table
    # dp[s] = min number of divisors to sum to s
    # parent[s] = the divisor that was added to reach s
    # prev_sum[s] = the sum before adding parent[s]
    
    INF = float('inf')
    dp = [INF] * (target + 1)
    dp[0] = 0
    
    parent = [-1] * (target + 1)
    prev_sum = [-1] * (target + 1)
    
    # Iterate through each divisor
    for d in valid_divisors:
        if time.time() > deadline:
            break
        # Update dp in reverse to ensure 0/1 knapsack property (distinct divisors)
        # We only care about sums up to target
        for s in range(target, d - 1, -1):
            if dp[s - d] + 1 < dp[s]:
                dp[s] = dp[s - d] + 1
                parent[s] = d
                prev_sum[s] = s - d
    
    if dp[target] == INF:
        # Fallback, though shouldn't happen for practical numbers
        # Greedy fallback
        remainder = target
        chosen = []
        for d in sorted(valid_divisors, reverse=True):
            if d <= remainder:
                chosen.append(d)
                remainder -= d
            if remainder == 0:
                break
        return chosen

    # Backtrack to find the selected divisors
    selected = []
    s = target
    while s > 0:
        d = parent[s]
        if d == -1:
            # Should not happen if dp[target] is valid
            break
        selected.append(d)
        s = prev_sum[s]
        
    return selected
