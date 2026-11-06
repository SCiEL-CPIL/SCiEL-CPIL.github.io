import time

def construct():
    n = 5040
    target = 4979
    
    # Generate divisors
    divisors = []
    for d in range(1, n + 1):
        if n % d == 0:
            divisors.append(d)
    
    m = len(divisors)
    
    # DP Table
    # dp[i][s] = min number of divisors from first i divisors to sum to s
    # We use a large number for infinity
    INF = 10**9
    # We only need the previous row to compute the current one, 
    # but for backtracking we need the choice.
    # Storing full 2D table is 60 * 5000 = 300k entries, which is fine.
    # Let's use lists of lists.
    dp = [[INF] * (target + 1) for _ in range(m + 1)]
    choice = [[False] * (target + 1) for _ in range(m + 1)]
    
    dp[0][0] = 0
    
    for i in range(1, m + 1):
        d = divisors[i - 1]
        for s in range(target + 1):
            # Option 1: Skip divisor i-1
            dp[i][s] = dp[i - 1][s]
            choice[i][s] = False
            
            # Option 2: Take divisor i-1
            if s >= d:
                if dp[i - 1][s - d] + 1 < dp[i][s]:
                    dp[i][s] = dp[i - 1][s - d] + 1
                    choice[i][s] = True
    
    min_count = dp[m][target]
    
    if min_count == INF:
        # Fallback if no solution found (should not happen for this instance)
        # Try a greedy approach as a last resort, though it might not be optimal
        remainder = target
        chosen = []
        for d in sorted(divisors, reverse=True):
            if d <= remainder:
                chosen.append(d)
                remainder -= d
        if remainder == 0:
            return chosen
        raise ValueError("No solution found")
    
    # Backtrack to find the selected divisors
    chosen = []
    s = target
    for i in range(m, 0, -1):
        if choice[i][s]:
            d = divisors[i - 1]
            chosen.append(d)
            s -= d
    
    # Verify
    assert len(set(chosen)) == len(chosen), "Not distinct"
    assert all(n % d == 0 for d in chosen), "Not all divisors of 5040"
    assert sum(chosen) == target, f"Sum mismatch: {sum(chosen)} != {target}"
    assert len(chosen) <= 256, "Too many divisors"
    
    return chosen
