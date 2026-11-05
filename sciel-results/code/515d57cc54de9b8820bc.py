import time

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill
    
    n = 5040
    target = 4979
    
    # Generate all divisors of n
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    
    # Filter: only divisors <= target can be used
    divisors = [d for d in divisors if d <= target]
    divisors.sort(reverse=True)  # Process larger divisors first
    
    N = len(divisors)
    
    # Quick greedy to get an upper bound on k
    remainder = target
    greedy_chosen = []
    for d in divisors:  # Already sorted descending
        if d <= remainder:
            greedy_chosen.append(d)
            remainder -= d
            if remainder == 0:
                break
    greedy_k = len(greedy_chosen) if remainder == 0 else N + 1
    
    # DP with 1D array and parent tracking
    # dp[s] = min number of divisors to sum to s
    # parent[s] = index of the last divisor added to reach sum s (or -1 if not reached)
    # We need to track which divisors were used, so we use a different approach
    
    INF = 10**9
    dp = [INF] * (target + 1)
    dp[0] = 0
    
    # parent[s] = (divisor_index, previous_sum) for backtracking
    parent = [(-1, -1)] * (target + 1)
    
    # Process each divisor
    for i, d in enumerate(divisors):
        # Update dp in reverse order to ensure each divisor is used at most once
        for s in range(target, d - 1, -1):
            if dp[s - d] + 1 < dp[s]:
                dp[s] = dp[s - d] + 1
                parent[s] = (i, s - d)
        
        # Time check
        if time.time() > deadline:
            break
    
    # Check if solution exists
    if dp[target] == INF:
        # Fallback: return greedy result if valid
        if remainder == 0:
            return greedy_chosen
        return []
    
    # Backtrack to find the divisors
    selected = []
    s = target
    while s > 0:
        div_idx, prev_s = parent[s]
        if div_idx == -1:
            break
        selected.append(divisors[div_idx])
        s = prev_s
    
    # Verify
    assert sum(selected) == target, f"Sum mismatch: {sum(selected)} != {target}"
    assert len(set(selected)) == len(selected), "Not distinct"
    for d in selected:
        assert n % d == 0, f"{d} is not a divisor of {n}"
    
    return selected
