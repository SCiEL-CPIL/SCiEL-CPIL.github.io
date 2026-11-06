import math
import bisect

def construct():
    limit = 10000
    # Sieve of Eratosthenes
    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i * i, limit + 1, i):
                is_prime[j] = False
    
    primes = [i for i, p in enumerate(is_prime) if p]
    N = len(primes)
    
    if N == 0:
        return []
    if N == 1:
        return primes
    
    # dp[i][j] = max length of valid sequence ending with primes[i], primes[j]
    dp = [[0] * N for _ in range(N)]
    # pred[i][j] = index k such that the best sequence ending at i, j came from k, i
    pred = [[-1] * N for _ in range(N)]
    
    # suf_max[j][k] = max(dp[c][j]) for c in [k, j-1]
    # suf_arg[j][k] = argmax c in [k, j-1] for dp[c][j]
    suf_max = [None] * N
    suf_arg = [None] * N
    
    for j in range(1, N):
        for i in range(j):
            gap = primes[j] - primes[i]
            target = primes[i] - gap
            # Find smallest k < i such that primes[k] >= target
            # This ensures primes[i] - primes[k] <= gap
            k_min = bisect.bisect_left(primes, target, 0, i)
            
            if k_min < i:
                # There is at least one valid predecessor k
                best_len = suf_max[i][k_min]
                best_k = suf_arg[i][k_min]
                dp[i][j] = best_len + 1
                pred[i][j] = best_k
            else:
                # No valid predecessor, sequence starts at i, j
                dp[i][j] = 2
                pred[i][j] = -1
        
        # Build suffix max and argmax structures for column j
        # These will be used when j becomes 'i' in future iterations
        sm = [0] * j
        sa = [-1] * j
        cur_max = 0
        cur_arg = -1
        for k in range(j - 1, -1, -1):
            val = dp[k][j]
            if val > cur_max:
                cur_max = val
                cur_arg = k
            sm[k] = cur_max
            sa[k] = cur_arg
        suf_max[j] = sm
        suf_arg[j] = sa
    
    # Find the maximum length and the ending indices
    max_len = 0
    end_i = 0
    end_j = 1
    for j in range(1, N):
        for i in range(j):
            if dp[i][j] > max_len:
                max_len = dp[i][j]
                end_i = i
                end_j = j
                
    # Reconstruct the path
    path = []
    i, j = end_i, end_j
    while j > 0:
        path.append(primes[j])
        k = pred[i][j]
        if k == -1:
            path.append(primes[i])
            break
        i, j = k, i
        
    path.reverse()
    return path
