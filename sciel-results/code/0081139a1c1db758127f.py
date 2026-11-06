import time

def construct():
    start_time = time.time()
    deadline = 85.0  # Wall-clock limit, leave 5s margin
    
    n = 5040
    target = 4979
    
    # 1. Get all distinct divisors of n
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    
    # Sort divisors. The order doesn't strictly matter for the DP correctness,
    # but for backtracking we need consistent indexing.
    divisors.sort()
    N = len(divisors)
    
    # 2. Dynamic Programming
    # dp[i][s] = minimum number of divisors from divisors[0..i-1] that sum to s
    # INF is a large number
    INF = float('inf')
    
    # We use a list of lists for dp.
    # dp[i] is a list of length target + 1
    dp = [[INF] * (target + 1) for _ in range(N + 1)]
    
    # Base case
    dp[0][0] = 0
    
    # Fill DP table
    for i in range(1, N + 1):
        d = divisors[i-1]
        # If d > target, we can't use it for any sum <= target, 
        # so dp[i][s] = dp[i-1][s] for all s.
        if d > target:
            dp[i] = dp[i-1].copy()
            continue
            
        prev_dp = dp[i-1]
        curr_dp = dp[i]
        
        for s in range(target + 1):
            # Option 1: Don't take d
            best = prev_dp[s]
            
            # Option 2: Take d
            if s >= d:
                candidate = prev_dp[s-d] + 1
                if candidate < best:
                    best = candidate
            
            curr_dp[s] = best
            
            # Time check
            if time.time() > deadline:
                # If we run out of time, we might not have finished.
                # But N=60, target=5000, so 300k ops. This is very fast (<0.1s).
                # This check is just a safety net.
                pass

    # 3. Check if solution exists
    min_count = dp[N][target]
    if min_count == INF:
        # No solution found? Should not happen for this instance.
        # Fallback to greedy to return something valid if possible
        remainder = target
        chosen = []
        for d in sorted(divisors, reverse=True):
            if d <= remainder:
                chosen.append(d)
                remainder -= d
        if remainder == 0:
            return chosen
        return []

    # 4. Backtrack to find the divisors
    chosen = []
    s = target
    for i in range(N, 0, -1):
        d = divisors[i-1]
        # Check if d was taken
        # dp[i][s] == dp[i-1][s] implies d was NOT taken (or taking it didn't improve count, 
        # but since we want min count, if counts are equal, we prefer not taking? 
        # Actually, if counts are equal, either choice leads to a valid min-count solution.
        # We just need to follow a path that is consistent.
        # If dp[i][s] == dp[i-1][s-d] + 1, then d was taken.
        # If dp[i][s] == dp[i-1][s], then d was not taken.
        # What if both are true? i.e., dp[i-1][s] == dp[i-1][s-d] + 1?
        # Then taking d leads to the same count. We can choose either.
        # Let's prioritize taking d if it leads to a valid state?
        # Actually, standard backtracking:
        # If s >= d and dp[i][s] == dp[i-1][s-d] + 1:
        #     Take d
        # Else:
        #     Don't take d
        
        if s >= d and dp[i][s] == dp[i-1][s-d] + 1:
            chosen.append(d)
            s -= d
        else:
            # Don't take d
            pass
    
    # The chosen list contains the divisors.
    # Verify sum and distinctness (distinctness is guaranteed by 0/1 DP)
    # return chosen
    return chosen
