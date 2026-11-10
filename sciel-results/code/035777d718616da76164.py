"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from math import isqrt
    import bisect

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
    # parent[i][j] = index h such that primes[h] is the predecessor of primes[i] in the optimal sequence ending with primes[i], primes[j]
    
    dp = [[2] * n for _ in range(n)]
    parent = [[-1] * n for _ in range(n)]
    
    best_len = 2
    best_pair = (0, 1)
    
    # Iterate i from 1 to n-2 (i is the second-to-last element)
    for i in range(1, n - 1):
        # We need to compute max dp[h][i] for h < i such that primes[i] - primes[h] <= primes[j] - primes[i]
        # Let Limit = primes[i] - (primes[j] - primes[i]) = 2*primes[i] - primes[j]
        # Condition: primes[h] >= Limit
        # Since primes are sorted, valid h form a contiguous range [h_min, i-1]
        # We precompute suffix maximums of dp[h][i] for h in [0, i-1]
        
        # suf_val[h] = max(dp[k][i] for k in [h, i-1])
        # suf_idx[h] = argmax(dp[k][i] for k in [h, i-1])
        
        suf_val = [0] * i
        suf_idx = [0] * i
        
        if i > 0:
            suf_val[i-1] = dp[i-1][i]
            suf_idx[i-1] = i-1
            
            for h in range(i-2, -1, -1):
                if dp[h][i] > suf_val[h+1]:
                    suf_val[h] = dp[h][i]
                    suf_idx[h] = h
                else:
                    suf_val[h] = suf_val[h+1]
                    suf_idx[h] = suf_idx[h+1]
        
        # For each j > i
        for j in range(i + 1, n):
            limit = 2 * primes[i] - primes[j]
            # Find smallest h in [0, i-1] such that primes[h] >= limit
            h_min = bisect.bisect_left(primes, limit, 0, i)
            
            if h_min < i:
                best_val = suf_val[h_min]
                best_h = suf_idx[h_min]
                dp[i][j] = 2 + best_val
                parent[i][j] = best_h
                
                if dp[i][j] > best_len:
                    best_len = dp[i][j]
                    best_pair = (i, j)
    
    # Backtrack to reconstruct the sequence
    if best_len == 2:
        return [primes[0], primes[1]]
        
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
