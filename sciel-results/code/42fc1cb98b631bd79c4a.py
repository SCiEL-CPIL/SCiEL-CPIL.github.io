import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Sieve for primes up to 10000
    limit = 10000
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    
    primes = [i for i, v in enumerate(is_prime) if v]
    N = len(primes)
    
    if N == 0:
        return []
        
    # dp[i][l] = min last gap for sequence of length l ending at i
    # parent[i][l] = j (previous index)
    # We use a large number for infinity
    INF = 10**9
    
    # Max possible length is N, but practically much smaller.
    # We can dynamically size the lists.
    # Start with length 1 for all
    dp = [[INF] * (N + 1) for _ in range(N)]
    parent = [[-1] * (N + 1) for _ in range(N)]
    
    for i in range(N):
        dp[i][1] = 0
        parent[i][1] = -1
        
    best_len = 1
    best_index = 0
    # Track max length achieved at each index to bound inner loop
    max_len_at = [1] * N
    
    for i in range(1, N):
        if time.time() > deadline:
            break
            
        pi = primes[i]
        max_len_at_i = 1
        
        for j in range(i):
            gap = pi - primes[j]
            
            # Only iterate l up to max_len_at[j]
            # dp[j][l] is non-decreasing in l, stop when dp[j][l] > gap
            j_max_l = max_len_at[j]
            
            for l in range(1, j_max_l + 1):
                if dp[j][l] > gap:
                    break
                    
                new_len = l + 1
                if gap < dp[i][new_len]:
                    dp[i][new_len] = gap
                    parent[i][new_len] = j
                    if new_len > max_len_at_i:
                        max_len_at_i = new_len
                        
        max_len_at[i] = max_len_at_i
        if max_len_at_i > best_len:
            best_len = max_len_at_i
            best_index = i
            
    # Reconstruct
    if best_len == 1:
        return [primes[best_index]]
        
    path = []
    curr_idx = best_index
    curr_len = best_len
    
    while curr_len > 0:
        path.append(primes[curr_idx])
        if curr_len == 1:
            break
        prev_idx = parent[curr_idx][curr_len]
        curr_idx = prev_idx
        curr_len -= 1
        
    path.reverse()
    return path
