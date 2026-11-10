import bisect
from math import isqrt

def construct():
    # Generate primes up to 10000
    limit = 10000
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, isqrt(limit) + 1):
        if sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = False
    primes = [i for i, is_prime in enumerate(sieve) if is_prime]
    N = len(primes)
    
    if N < 2:
        return primes[:]

    # dp[i][j] stores the length of the longest valid sequence ending with primes[i], primes[j]
    # We use a list of lists.
    dp = [[0] * N for _ in range(N)]
    prev = [[-1] * N for _ in range(N)]
    
    for i in range(N):
        for j in range(i + 1, N):
            dp[i][j] = 2

    best_len = 2
    best_i = 0
    best_j = 1

    for j in range(1, N):
        # Build suffix max structure for dp[?][j]
        # We need max dp[m][j] for m in [i_min, j-1]
        suf_len = [0] * j
        suf_idx = [-1] * j
        
        current_max_val = 0
        current_max_idx = -1
        
        for m in range(j - 1, -1, -1):
            if dp[m][j] > current_max_val:
                current_max_val = dp[m][j]
                current_max_idx = m
            suf_len[m] = current_max_val
            suf_idx[m] = current_max_idx

        for k in range(j + 1, N):
            gap = primes[k] - primes[j]
            L = 2 * primes[j] - primes[k]
            
            i_min = bisect.bisect_left(primes, L)
            
            if i_min < j:
                best_prev_len = suf_len[i_min]
                best_prev_idx = suf_idx[i_min]
                
                if best_prev_len > 0:
                    new_len = best_prev_len + 1
                    if new_len > dp[j][k]:
                        dp[j][k] = new_len
                        prev[k][j] = best_prev_idx
                        
                        if new_len > best_len:
                            best_len = new_len
                            best_i = j
                            best_j = k

    # Reconstruct sequence
    if best_len == 2:
        return [primes[best_i], primes[best_j]]

    indices = [best_j, best_i]
    curr_j = best_j
    curr_i = best_i
    
    while True:
        prev_i = prev[curr_j][curr_i]
        if prev_i == -1:
            break
        indices.append(prev_i)
        curr_j = curr_i
        curr_i = prev_i
        
    indices.reverse()
    return [primes[idx] for idx in indices]
