import time
import math
import numpy as np

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

    INF = 10**9

    # dp[i][l] = min last gap for sequence of length l ending at prime i
    # max_len[i] = max l such that dp[i][l] < INF
    dp = [[INF] * (N + 1) for _ in range(N)]
    parent = [[-1] * (N + 1) for _ in range(N)]
    max_len = [0] * N

    for i in range(N):
        dp[i][1] = 0
        max_len[i] = 1

    best_len = 1
    best_index = 0

    for i in range(1, N):
        if time.time() > deadline:
            break

        pi = primes[i]
        max_len_at_i = 1

        for j in range(i):
            gap = pi - primes[j]
            dp_j = dp[j]
            dp_i = dp[i]
            par_i = parent[i]
            ml_j = max_len[j]

            for l in range(1, ml_j + 1):
                if dp_j[l] > gap:
                    break
                new_len = l + 1
                if gap < dp_i[new_len]:
                    dp_i[new_len] = gap
                    par_i[new_len] = j
                    if new_len > max_len_at_i:
                        max_len_at_i = new_len

        max_len[i] = max_len_at_i
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
