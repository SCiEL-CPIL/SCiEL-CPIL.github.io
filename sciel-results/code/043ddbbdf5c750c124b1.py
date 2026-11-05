import time

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill

    n = 5040
    target = 4979

    # Generate all distinct positive divisors of 5040
    divisors = []
    for d in range(1, n + 1):
        if n % d == 0:
            divisors.append(d)
    
    # DP: dp[s] = minimum number of divisors to sum to s
    INF = float('inf')
    dp = [INF] * (target + 1)
    dp[0] = 0
    parent = [(-1, -1)] * (target + 1)  # (prev_sum, divisor)
    
    for d in divisors:
        for s in range(target, d - 1, -1):
            if dp[s - d] + 1 < dp[s]:
                dp[s] = dp[s - d] + 1
                parent[s] = (s - d, d)
    
    if dp[target] == INF:
        # Fallback: greedy
        divisors_desc = sorted(divisors, reverse=True)
        remainder = target
        chosen = []
        for d in divisors_desc:
            if d <= remainder:
                chosen.append(d)
                remainder -= d
            if remainder == 0:
                break
        assert remainder == 0
        return chosen
    
    # Reconstruct
    solution = []
    s = target
    while s > 0:
        prev_s, d = parent[s]
        solution.append(d)
        s = prev_s
    
    solution.reverse()
    
    # Verify
    assert len(set(solution)) == len(solution), "Not distinct"
    assert all(5040 % d == 0 for d in solution), "Not divisors"
    assert sum(solution) == target, f"Sum mismatch: {sum(solution)}"
    assert len(solution) <= 256, "Too many divisors"
    
    return solution
