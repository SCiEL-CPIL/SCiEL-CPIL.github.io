import time

def _get_divisors(n):
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    return divisors

def construct():
    start_time = time.time()
    timeout = 50.0
    
    n = 5040
    target = 4979
    
    divisors = _get_divisors(n)
    N = len(divisors)
    
    # dp[s] = min count to reach sum s
    INF = float('inf')
    dp = [INF] * (target + 1)
    dp[0] = 0
    
    # parent[s] = (prev_sum, divisor_index) to reconstruct
    parent_sum = [-1] * (target + 1)
    parent_div_idx = [-1] * (target + 1)
    
    # Iterate through divisors
    for i, d in enumerate(divisors):
        # Update dp backwards
        for s in range(target, d - 1, -1):
            if dp[s - d] + 1 < dp[s]:
                dp[s] = dp[s - d] + 1
                parent_sum[s] = s - d
                parent_div_idx[s] = i
        
        if time.time() - start_time > timeout:
            break

    if dp[target] == INF:
        # Fallback: Greedy
        remainder = target
        chosen = []
        for d in sorted(divisors, reverse=True):
            if d <= remainder:
                chosen.append(d)
                remainder -= d
                if remainder == 0:
                    break
        return chosen

    # Reconstruct
    result = []
    s = target
    while s > 0:
        idx = parent_div_idx[s]
        result.append(divisors[idx])
        s = parent_sum[s]
        
    return result
