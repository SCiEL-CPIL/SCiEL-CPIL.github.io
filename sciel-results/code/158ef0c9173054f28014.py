import time
import math
import bisect

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin

    # Sieve of Eratosthenes to get primes up to 10000
    limit = 10000
    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(math.sqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i * i, limit + 1, i):
                is_prime[j] = False
    primes = [p for p in range(2, limit + 1) if is_prime[p]]
    n = len(primes)
    
    # dp[i][g] = (length, prev_j, prev_g)
    dp = [{} for _ in range(n)]
    
    # sorted_dp[i] = list of (gap, max_length, best_g_in_dp_i)
    sorted_dp = [[] for _ in range(n)]
    
    best_len = 1
    best_i = 0
    best_g = 0
    
    for i in range(n):
        if time.time() > deadline:
            break
            
        for j in range(i):
            g = primes[i] - primes[j]
            
            best_prev_len = 1
            best_prev_g = 0
            
            if sorted_dp[j]:
                idx = bisect.bisect_right(sorted_dp[j], (g,))
                if idx > 0:
                    _, max_len, best_g_j = sorted_dp[j][idx-1]
                    if max_len > best_prev_len:
                        best_prev_len = max_len
                        best_prev_g = best_g_j
            
            new_len = best_prev_len + 1
            
            if g not in dp[i] or new_len > dp[i][g][0]:
                dp[i][g] = (new_len, j, best_prev_g)
        
        if dp[i]:
            items = list(dp[i].items())
            items.sort(key=lambda x: x[0])
            
            max_len_so_far = 0
            best_g_so_far = 0
            for g, (L, pj, pg) in items:
                if L > max_len_so_far:
                    max_len_so_far = L
                    best_g_so_far = g
                sorted_dp[i].append((g, max_len_so_far, best_g_so_far))
            
            for g, (L, pj, pg) in dp[i].items():
                if L > best_len:
                    best_len = L
                    best_i = i
                    best_g = g
    
    seq = []
    i = best_i
    g = best_g
    while i != -1:
        seq.append(primes[i])
        if g == 0:
            break
        L, pj, pg = dp[i][g]
        i = pj
        g = pg
    seq.reverse()
    return seq
