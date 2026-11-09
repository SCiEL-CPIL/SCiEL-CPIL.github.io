import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0 # Safety margin
    
    n = 5040
    target = 4979
    
    # Generate divisors
    divs = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
    divs.sort()
    
    M = len(divs)
    
    # DP Table
    # dp[i][s] = min divisors from divs[0..i-1] to sum to s
    # Use a large number for infinity
    INF = 10**9
    dp = [[INF] * (target + 1) for _ in range(M + 1)]
    dp[0][0] = 0
    
    # Fill DP table
    for i in range(1, M + 1):
        d = divs[i-1]
        if d > target:
            # If divisor is larger than target, we can't use it for any sum <= target
            # But we still need to copy previous row?
            # Actually, if d > target, it can't be part of any sum <= target.
            # So dp[i][s] = dp[i-1][s] for all s.
            # We can just copy the row or skip update.
            # Since we are using a 2D list, we can do:
            for s in range(target + 1):
                dp[i][s] = dp[i-1][s]
            continue
            
        for s in range(target + 1):
            # Option 1: Don't pick d
            best = dp[i-1][s]
            # Option 2: Pick d
            if s >= d:
                val = dp[i-1][s-d] + 1
                if val < best:
                    best = val
            dp[i][s] = best
            
        # Time check (optional, but good practice)
        if time.time() > deadline:
             # Should not happen, but if it does, we might have partial result?
             # No, we need the full table to backtrack.
             # Given speed, this is unlikely.
             pass

    if dp[M][target] == INF:
        # Should not happen
        return []
        
    # Backtrack
    res = []
    s = target
    i = M
    while i > 0 and s > 0:
        d = divs[i-1]
        if s >= d and dp[i-1][s-d] + 1 == dp[i][s]:
            res.append(d)
            s -= d
            i -= 1
        else:
            i -= 1
            
    return res
