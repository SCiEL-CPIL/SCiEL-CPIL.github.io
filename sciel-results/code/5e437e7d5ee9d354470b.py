import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    
    n = 5040
    target = 4979
    
    # Generate all divisors of n
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    
    # Sort divisors in descending order to try larger divisors first (may find solutions faster)
    divisors.sort(reverse=True)
    num_divisors = len(divisors)
    
    # Precompute suffix sums for pruning
    # suffix_sum[i] = sum of divisors[i:]
    suffix_sum = [0] * (num_divisors + 1)
    for i in range(num_divisors - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + divisors[i]
    
    # dp[s] = minimum number of divisors to reach sum s using some subset of processed divisors
    INF = float('inf')
    dp = [INF] * (target + 1)
    parent = [-1] * (target + 1)
    prev_sum = [-1] * (target + 1)
    
    dp[0] = 0
    
    # Process divisors in descending order
    for i, d in enumerate(divisors):
        if time.time() > deadline:
            break
        
        # Pruning: if even using all remaining divisors can't reach target from any current state, skip
        # But this is hard to check per-state, so let's just do the standard DP
        
        for s in range(target, d - 1, -1):
            if dp[s - d] + 1 < dp[s]:
                dp[s] = dp[s - d] + 1
                parent[s] = i
                prev_sum[s] = s - d
    
    if dp[target] == INF:
        remaining = target
        chosen = []
        for d in divisors:  # Already in descending order
            if d <= remaining:
                chosen.append(d)
                remaining -= d
                if remaining == 0:
                    return chosen
        return chosen
    
    current_sum = target
    chosen = []
    while current_sum > 0:
        idx = parent[current_sum]
        chosen.append(divisors[idx])
        current_sum = prev_sum[current_sum]
        
    return chosen
