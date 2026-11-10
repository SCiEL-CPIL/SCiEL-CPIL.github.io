"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from math import isqrt
    
    # Generate all primes <= 10000
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, isqrt(n) + 1):
            if is_prime[i]:
                for j in range(i * i, n + 1, i):
                    is_prime[j] = False
        return [i for i in range(2, n + 1) if is_prime[i]]
    
    primes = sieve(10000)
    n = len(primes)
    
    if n <= 2:
        return primes
    
    # dp[i][j] = length of longest valid subsequence ending with primes[i], primes[j] (i < j)
    # We'll use a 2D list, but optimize the transition
    # dp[i][j] = 2 + max(dp[h][i] for h < i where primes[i] - primes[h] <= primes[j] - primes[i])
    
    # To make this efficient, for each i, we process j in increasing order
    # and maintain the best dp[h][i] for h satisfying the constraint
    
    # dp table: dp[i][j] for i < j
    # Initialize all to 2
    dp = [[2] * n for _ in range(n)]
    
    best_len = 2
    best_end = (0, 1)  # (i, j) indices
    
    # For each middle index i, compute dp[h][i] for all h < i, 
    # then for each j > i, compute dp[i][j]
    
    # Actually, let's do it differently:
    # For each i (as the second-to-last), and for each j > i (as last),
    # dp[i][j] = 2 + max{dp[h][i] : h < i, primes[i] - primes[h] <= primes[j] - primes[i]}
    
    # Precompute for each i: an array best_for_i where for each possible gap g,
    # we store the max dp[h][i] for h < i with primes[i] - primes[h] <= g
    
    # Since primes are sorted, for fixed i, as h decreases, primes[i] - primes[h] increases.
    # So for fixed i, let's create a list of (gap, dp[h][i]) for h = 0 to i-1, sorted by gap.
    
    # For efficiency, for each i, we'll process j from i+1 to n-1, and maintain a pointer
    # into the h values.
    
    # Let's restructure: for each i, we have h values 0..i-1. 
    # For each h < i, gap_h = primes[i] - primes[h]
    # We want for each j > i: max dp[h][i] where gap_h <= primes[j] - primes[i]
    
    # Precompute for each i: sort h by gap_h (which is decreasing as h increases since primes are increasing)
    # Actually gap_h = primes[i] - primes[h], so as h increases, gap_h decreases.
    
    # For fixed i, let's create array of (gap, dp[h][i]) for h=0..i-1
    # Since primes[h] is increasing in h, gap_h is decreasing in h.
    # So for h = i-1, i-2, ..., 0, gap_h is increasing.
    
    # Let me just use a different approach: for each i, build a list of (gap, dp[h][i]) 
    # and prefix max.
    
    # dp[i][j] for i < j
    # We'll compute iteratively.
    
    # Initialize: all dp[i][j] = 2 for i < j
    # Then for each i from 1 to n-2:
    #   For each j from i+1 to n-1:
    #     dp[i][j] = 2 + max(dp[h][i] for h < i, primes[i] - primes[h] <= primes[j] - primes[i])
    
    # To compute the max efficiently for fixed i:
    #   Create list of (gap_h, dp[h][i]) for h=0..i-1, where gap_h = primes[i] - primes[h]
    #   Sort by gap_h (ascending)
    #   Compute prefix max of dp values
    #   For each j, binary search for the largest gap_h <= primes[j] - primes[i], get prefix max
    
    # This is O(n^2 log n) which is ~1229^2 * log(1229) ≈ 1.5M * 11 ≈ 16M, feasible.
    
    import bisect
    
    # dp[i][j] for i < j
    dp = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            dp[i][j] = 2
    
    best_len = 2
    best_pair = (0, 1)
    
    for i in range(1, n - 1):  # i is the second-to-last prime index
        # Build list of (gap, dp[h][i]) for h = 0 to i-1
        gap_dp = []
        for h in range(i):
            gap = primes[i] - primes[h]
            gap_dp.append((gap, dp[h][i]))
        
        # Sort by gap ascending
        gap_dp.sort(key=lambda x: x[0])
        
        # Build prefix max of dp values
        gaps = [x[0] for x in gap_dp]
        prefix_max = [0] * len(gap_dp)
        prefix_max[0] = gap_dp[0][1]
        for k in range(1, len(gap_dp)):
            prefix_max[k] = max(prefix_max[k - 1], gap_dp[k][1])
        
        # For each j > i, compute dp[i][j]
        for j in range(i + 1, n):
            required_gap = primes[j] - primes[i]
            # Find rightmost index in gaps where gaps[idx] <= required_gap
            idx = bisect.bisect_right(gaps, required_gap) - 1
            if idx >= 0:
                dp[i][j] = 2 + prefix_max[idx]
            # else dp[i][j] stays 2
            if dp[i][j] > best_len:
                best_len = dp[i][j]
                best_pair = (i, j)
    
    # Reconstruct the sequence
    if best_len == 2:
        return [primes[0], primes[1]]
    
    # We need to backtrack. dp[i][j] tells us the length, but we need to know which h was used.
    # Let's store the argmax as well.
    
    # Redo with backtracking info
    dp = [[0] * n for _ in range(n)]
    parent = [[-1] * n for _ in range(n)]  # parent[i][j] = h that achieved dp[i][j]
    for i in range(n):
        for j in range(i + 1, n):
            dp[i][j] = 2
    
    best_len = 2
    best_pair = (0, 1)
    
    for i in range(1, n - 1):
        gap_dp = []
        for h in range(i):
            gap = primes[i] - primes[h]
            gap_dp.append((gap, dp[h][i], h))
        
        gap_dp.sort(key=lambda x: x[0])
        
        gaps = [x[0] for x in gap_dp]
        prefix_max = [0] * len(gap_dp)
        prefix_max_idx = [0] * len(gap_dp)
        prefix_max[0] = gap_dp[0][1]
        prefix_max_idx[0] = gap_dp[0][2]
        for k in range(1, len(gap_dp)):
            if gap_dp[k][1] > prefix_max[k - 1]:
                prefix_max[k] = gap_dp[k][1]
                prefix_max_idx[k] = gap_dp[k][2]
            else:
                prefix_max[k] = prefix_max[k - 1]
                prefix_max_idx[k] = prefix_max_idx[k - 1]
        
        for j in range(i + 1, n):
            required_gap = primes[j] - primes[i]
            idx = bisect.bisect_right(gaps, required_gap) - 1
            if idx >= 0:
                dp[i][j] = 2 + prefix_max[idx]
                parent[i][j] = prefix_max_idx[idx]
            else:
                parent[i][j] = -1
            if dp[i][j] > best_len:
                best_len = dp[i][j]
                best_pair = (i, j)
    
    # Backtrack
    seq = []
    i, j = best_pair
    while i != -1:
        seq.append(primes[j])
        h = parent[i][j]
        if h == -1:
            break
        j = i
        i = h
    seq.append(primes[i])
    seq.reverse()
    
    return seq
# EVOLVE-BLOCK-END
