"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
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
    divisors.sort(reverse=True)
    
    m = len(divisors)
    
    # Dynamic programming for minimum cardinality subset sum
    # dp[s] = minimum number of divisors needed to sum to s, or -1 if impossible
    # We also track the actual set of divisors used
    
    # Since target is 4979 and divisors are at most 5040, we can use a dict-based DP
    # dp maps (sum, last_index) -> (count, path) but that's too large
    # Instead, use a simpler approach: for each divisor, update a dp table
    
    # dp[s] = minimum number of divisors to reach sum s
    # parent[s] = (previous_sum, divisor_used) to reconstruct the path
    
    # But we need distinct divisors, so we process them one by one
    # dp[i][s] = min count using divisors[0..i-1] to sum to s
    # This is O(m * target) which is 60 * 4979 = ~300k states, very feasible
    
    INF = float('inf')
    
    # dp[s] = (min_count, last_divisor_index) 
    # We'll do it iteratively: for each divisor, update dp
    # dp[s] = minimum number of divisors from the processed set that sum to s
    
    # Use a dictionary for sparsity, but target is small enough for an array
    max_sum = target
    
    # dp[s] = min number of divisors to reach sum s
    dp = [INF] * (max_sum + 1)
    dp[0] = 0
    
    # parent[s] = (prev_sum, divisor_index) to trace back
    # But we need to know which divisor was used at each step
    # Store: for each sum s, the last divisor index used and the previous sum
    prev_sum = [-1] * (max_sum + 1)
    prev_div_idx = [-1] * (max_sum + 1)
    
    for idx, d in enumerate(divisors):
        if time.time() > deadline:
            break
        # Process in reverse to avoid using the same divisor twice
        for s in range(max_sum, d - 1, -1):
            if dp[s - d] + 1 < dp[s]:
                dp[s] = dp[s - d] + 1
                prev_sum[s] = s - d
                prev_div_idx[s] = idx
    
    if dp[target] == INF:
        # Fallback: greedy
        remainder = target
        chosen = []
        for d in divisors:
            if d <= remainder:
                chosen.append(d)
                remainder -= d
            if remainder == 0:
                break
        return chosen
    
    # Reconstruct the solution
    chosen = []
    s = target
    while s > 0:
        idx = prev_div_idx[s]
        chosen.append(divisors[idx])
        s = prev_sum[s]
    
    return chosen

# EVOLVE-BLOCK-END
