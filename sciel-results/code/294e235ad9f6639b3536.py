def _get_divisors(n):
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    return divisors

def construct():
    n = 5040
    target = 4979
    
    divisors = _get_divisors(n)
    N = len(divisors)
    
    INF = float('inf')
    dp = [INF] * (target + 1)
    dp[0] = 0
    
    # parent[s] stores the index of the divisor used to reach sum s
    # and the previous sum is implicitly s - divisors[idx]
    parent_div_idx = [-1] * (target + 1)
    
    # Iterate through divisors
    # We don't need to sort for optimality of count, but it's good practice
    # for deterministic output if there are ties, though the problem
    # accepts any valid minimal set.
    for i, d in enumerate(divisors):
        # Update dp backwards to ensure 0/1 knapsack property
        # (each divisor used at most once)
        for s in range(target, d - 1, -1):
            if dp[s - d] + 1 < dp[s]:
                dp[s] = dp[s - d] + 1
                parent_div_idx[s] = i
        
    if dp[target] == INF:
        # This should not happen for a practical number like 5040
        # with a target less than the sum of all divisors, but handle just in case
        return []

    # Reconstruct the subset
    result = []
    s = target
    while s > 0:
        idx = parent_div_idx[s]
        if idx == -1:
            break # Safety check, should not occur if dp[target] != INF
        d = divisors[idx]
        result.append(d)
        s -= d
        
    return result
