"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    n = 5040
    target = 4979
    
    # Generate divisors
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    
    # Sort descending
    divisors.sort(reverse=True)
    num_divs = len(divisors)
    
    INF = 10**9
    
    # dp[i][s] = min number of divisors from divisors[i:] to sum to s
    dp = [[INF] * (target + 1) for _ in range(num_divs + 1)]
    dp[num_divs][0] = 0
    
    for i in range(num_divs - 1, -1, -1):
        d = divisors[i]
        # Optimization: if d > target, we can't pick it for any s <= target
        # But divisors are sorted descending, so if d > target, all subsequent i' < i have d' > d > target?
        # No, sorted descending means d_0 is largest.
        # If d_i > target, then for all s in [0, target], s < d_i, so we can't pick it.
        # So dp[i][s] = dp[i+1][s].
        if d > target:
            for s in range(target + 1):
                dp[i][s] = dp[i+1][s]
            continue
            
        for s in range(target + 1):
            # Option 1: Don't pick d
            best = dp[i + 1][s]
            
            # Option 2: Pick d
            if s >= d:
                val = dp[i + 1][s - d] + 1
                if val < best:
                    best = val
            
            dp[i][s] = best
    
    if dp[0][target] == INF:
        # Fallback to greedy
        remainder = target
        chosen = []
        for d in divisors:
            if d <= remainder:
                chosen.append(d)
                remainder -= d
        return chosen
        
    result = []
    i = 0
    s = target
    while i < num_divs:
        d = divisors[i]
        if s >= d and dp[i][s] == dp[i + 1][s - d] + 1:
            result.append(d)
            s -= d
        i += 1
        
    return result
# EVOLVE-BLOCK-END
