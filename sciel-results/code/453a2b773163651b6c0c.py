"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import bisect

def construct():
    start = time.time()
    deadline = start + 50
    
    n_max = 10000
    is_prime = [True] * (n_max + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n_max**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n_max + 1, i):
                is_prime[j] = False
    primes = [p for p in range(2, n_max + 1) if is_prime[p]]
    n = len(primes)
    
    if n <= 2:
        return primes
    
    dp = [[0] * n for _ in range(n)]
    prev = [[-1] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(i + 1, n):
            dp[i][j] = 2
    
    best_len = 2
    best_j, best_k = 0, 1
    
    for j in range(1, n):
        if time.time() > deadline:
            break
        suff_val = [0] * (j + 1)
        suff_arg = [-1] * (j + 1)
        for i in range(j - 1, -1, -1):
            if dp[i][j] >= suff_val[i + 1]:
                suff_val[i] = dp[i][j]
                suff_arg[i] = i
            else:
                suff_val[i] = suff_val[i + 1]
                suff_arg[i] = suff_arg[i + 1]
        
        for k in range(j + 1, n):
            gap_jk = primes[k] - primes[j]
            threshold = primes[j] - gap_jk
            lo = bisect.bisect_left(primes, threshold)
            if lo >= j:
                dp[j][k] = 2
                prev[j][k] = -1
            else:
                bv = suff_val[lo]
                ba = suff_arg[lo]
                if bv > 0:
                    dp[j][k] = bv + 1
                    prev[j][k] = ba
                else:
                    dp[j][k] = 2
                    prev[j][k] = -1
            
            if dp[j][k] > best_len:
                best_len = dp[j][k]
                best_j, best_k = j, k
    
    if best_len <= 2:
        return primes[:2]
    
    seq = [primes[best_k], primes[best_j]]
    j, k = best_j, best_k
    while len(seq) < best_len:
        i = prev[j][k]
        if i < 0:
            break
        seq.append(primes[i])
        k = j
        j = i
    
    seq.reverse()
    return seq
# EVOLVE-BLOCK-END
