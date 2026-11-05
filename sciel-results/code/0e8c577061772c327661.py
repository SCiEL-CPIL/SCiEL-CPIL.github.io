import time

def construct():
    n = 5040
    target = 4979
    
    # Calculate divisors
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    
    N = len(divisors)
    
    # dp[s] = (min_count, bitmask) for sum s
    INF = float('inf')
    dp_count = [INF] * (target + 1)
    dp_mask = [0] * (target + 1)
    
    dp_count[0] = 0
    dp_mask[0] = 0
    
    start_time = time.time()
    deadline = start_time + 50.0
    
    # Process divisors in descending order (larger first) - this doesn't change
    # the optimal count but may help with tie-breaking toward fewer large divisors
    divisors_sorted = sorted(divisors, reverse=True)
    
    for i, d in enumerate(divisors_sorted):
        if time.time() > deadline:
            break
        
        # 0/1 knapsack: iterate backwards
        for s in range(target, d - 1, -1):
            if dp_count[s - d] + 1 < dp_count[s]:
                dp_count[s] = dp_count[s - d] + 1
                dp_mask[s] = dp_mask[s - d] | (1 << i)
    
    if dp_count[target] == INF:
        # Fallback: greedy (should not happen for practical number)
        remainder = target
        chosen = []
        for d in divisors_sorted:
            if d <= remainder:
                chosen.append(d)
                remainder -= d
                if remainder == 0:
                    break
        return chosen
    
    # Reconstruct using bitmask
    result = []
    for i in range(N):
        if (dp_mask[target] >> i) & 1:
            result.append(divisors_sorted[i])
    
    # Verify: all are distinct divisors of 5040, sum to target
    assert len(set(result)) == len(result), "Divisors must be distinct"
    assert all(n % d == 0 for d in result), "All must be divisors of 5040"
    assert sum(result) == target, f"Sum must be {target}, got {sum(result)}"
    assert len(result) <= 256, "At most 256 divisors"
    
    return result
