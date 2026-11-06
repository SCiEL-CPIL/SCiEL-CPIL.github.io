import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # safety margin before 60s limit
    
    n = 5040
    target = 4979
    
    # Get all positive divisors of n
    divisors = [i for i in range(1, n + 1) if n % i == 0]
    # Sort in descending order to prioritize larger divisors (fewer terms)
    divisors.sort(reverse=True)
    
    num_divs = len(divisors)
    
    # DP: dp[s] = min number of divisors to sum to s
    # 1D DP for space efficiency
    INF = float('inf')
    dp = [INF] * (target + 1)
    dp[0] = 0
    
    # prev[s] stores the index of the last divisor added to achieve dp[s]
    # For reconstruction: we need to know which divisor was added last
    # We'll use a 2D-like approach but with careful tracking
    # Actually, let's use a different reconstruction method:
    # Store parent pointers: for each state (i, s), store whether we took divisors[i-1]
    # But 2D DP is fine: 60 * 5000 = 300k entries, very small
    
    # Let's use 2D DP for clean reconstruction
    dp = [[INF] * (target + 1) for _ in range(num_divs + 1)]
    dp[0][0] = 0
    
    for i in range(1, num_divs + 1):
        d = divisors[i - 1]
        prev_row = dp[i - 1]
        curr_row = dp[i]
        for s in range(target + 1):
            # Option 1: Don't take divisors[i-1]
            curr_row[s] = prev_row[s]
            # Option 2: Take divisors[i-1] if s >= d
            if s >= d:
                val = prev_row[s - d] + 1
                if val < curr_row[s]:
                    curr_row[s] = val
    
    min_count = dp[num_divs][target]
    
    if min_count == INF:
        return []
    
    # Reconstruct the solution
    result = []
    s = target
    for i in range(num_divs, 0, -1):
        if dp[i][s] != dp[i - 1][s]:
            # We took divisors[i-1]
            result.append(divisors[i - 1])
            s -= divisors[i - 1]
    
    # Verify
    assert sum(result) == target, f"Sum mismatch: {sum(result)} != {target}"
    assert len(set(result)) == len(result), "Non-distinct divisors found"
    for d in result:
        assert n % d == 0, f"{d} is not a divisor of {n}"
    
    return sorted(result, reverse=True)
