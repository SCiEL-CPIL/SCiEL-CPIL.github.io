import time

def _get_divisors(n):
    """Generate all divisors of n, sorted in descending order."""
    divs = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
    divs.sort(reverse=True)
    return divs

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    
    n = 5040
    target = 4979
    
    divisors = _get_divisors(n)
    divisors = [d for d in divisors if d <= target]
    num_divs = len(divisors)
    
    if num_divs == 0:
        return []
    
    # DP approach: dp[s] = minimum number of divisors to achieve sum s
    # parent[s] = (prev_sum, divisor_index) to reconstruct the solution
    INF = float('inf')
    dp = [INF] * (target + 1)
    dp[0] = 0
    parent = [(-1, -1)] * (target + 1)  # (prev_sum, divisor_index)
    
    # Process each divisor (0/1 knapsack, reverse order for s)
    for idx, d in enumerate(divisors):
        if time.time() > deadline:
            break
        # Update in reverse to ensure each divisor used at most once
        for s in range(target, d - 1, -1):
            if dp[s - d] + 1 < dp[s]:
                dp[s] = dp[s - d] + 1
                parent[s] = (s - d, idx)
    
    if dp[target] == INF:
        # Fallback: use a known valid solution
        found = [2520, 1680, 560, 210, 9]
    else:
        # Reconstruct the solution
        found = []
        s = target
        while s > 0:
            prev_s, idx = parent[s]
            found.append(divisors[idx])
            s = prev_s
    
    assert sum(found) == target
    assert len(set(found)) == len(found)
    assert all(n % d == 0 for d in found)
    assert len(found) <= 256
    
    return found
