import time

def construct():
    n = 5040
    target = 4979
    
    # Generate divisors of 5040
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    
    # Sort divisors (order doesn't strictly matter for DP correctness, 
    # but consistent ordering helps with determinism)
    divisors.sort()
    
    # DP array: dp[s] = minimum number of divisors to sum to s
    # Initialize with a large number
    INF = 10**9
    dp = [INF] * (target + 1)
    dp[0] = 0
    
    # Parent array to reconstruct the solution
    # parent[s] stores the last divisor added to reach sum s
    parent = [-1] * (target + 1)
    
    # 0/1 Knapsack style DP
    # We iterate through each divisor and update dp for sums from target down to divisor
    for d in divisors:
        for s in range(target, d - 1, -1):
            if dp[s - d] + 1 < dp[s]:
                dp[s] = dp[s - d] + 1
                parent[s] = d
    
    # Check if a solution exists
    if dp[target] == INF:
        # Fallback: The problem guarantees a solution exists for this specific instance.
        # If it didn't, we would need to handle it, but the task implies a finite practical number instance.
        return []
    
    # Reconstruct the set of divisors
    chosen = []
    s = target
    while s > 0:
        d = parent[s]
        if d == -1:
            # This should not happen if dp[target] is valid
            break
        chosen.append(d)
        s -= d
    
    # The chosen list contains the divisors in reverse order of addition
    # The order does not matter for the sum, but we can sort for cleanliness
    # The task asks for "at most 256 distinct positive divisors", order is not specified.
    return chosen
