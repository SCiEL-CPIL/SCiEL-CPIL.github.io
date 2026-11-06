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

    # dp[i][l] = min last gap for sequence of length l ending at i
    # parent[i][l] = j
    # We use a list of lists.
    dp = [[INF] * (N + 1) for _ in range(N)]
    parent = [[-1] * (N + 1) for _ in range(N)]

    for i in range(N):
        dp[i][1] = 0
        parent[i][1] = -1

    best_len = 1
    best_index = 0

    # Optimization: Precompute gaps? No, N^2 gaps is fine.
    # We will use vectorized operations where possible, but the inner loop
    # over l is small, so we stick to the structure but try to optimize
    # the inner loop slightly by breaking early.

    # To speed up, we can precompute the gaps for each i, j?
    # Or just compute on the fly.
    
    # Let's try to use the fact that dp[j][l] is non-decreasing.
    # We already do that.
    
    # One improvement: if we find a sequence of length L, we can prune
    # searches that cannot beat it?
    # Actually, the current DP finds the global optimum for non-decreasing gaps.
    # Is there a faster way?
    # The complexity is O(N^2 * L_max). L_max is small (~20-30?).
    # N=1229. N^2 = 1.5M. 1.5M * 30 = 45M ops. This should be fast in Python?
    # 45M ops might take ~5-10s in Python.
    # The deadline is 55s. So it should finish.
    # The current score is 1.3243.
    # Let's see if we can improve the constant factor.
    
    # We can use numpy for the gap calculation?
    # gap = pi - primes[j]
    # If we vectorize the j loop?
    # For a fixed i, gaps = primes[i] - primes[:i]
    # Then for each j, we need to find max l such that dp[j][l] <= gaps[j].
    # This is hard to vectorize because dp is a 2D array and the condition depends on l.
    
    # Let's stick to the loop but optimize the inner loop.
    
    for i in range(1, N):
        if time.time() > deadline:
            break
            
        pi = primes[i]
        max_len_at_i = 1
        
        # We can iterate j in reverse? No, order doesn't matter for correctness,
        # only for potential early termination if we had a greedy bound, which we don't.
        
        for j in range(i):
            gap = pi - primes[j]
            
            # Iterate l
            # We know dp[j][1] = 0.
            # If gap >= 0, we can update dp[i][2].
            
            # To speed up, we can find the max l such that dp[j][l] <= gap
            # Since dp[j][l] is sorted, we can binary search?
            # But L is small, linear scan is faster.
            
            # We can unroll the first few steps?
            # l=1: dp[j][1]=0 <= gap (always true for j<i)
            # so dp[i][2] = min(dp[i][2], gap)
            
            # l=2: if dp[j][2] <= gap, update dp[i][3]
            # ...
            
            # Let's just use the loop.
            # One micro-optimization: store dp[j] in a local variable.
            dp_j = dp[j]
            dp_i = dp[i]
            par_i = parent[i]
            
            # We know dp_j[1] = 0.
            # If gap < dp_i[2]:
            #   dp_i[2] = gap
            #   par_i[2] = j
            #   if 2 > max_len_at_i: max_len_at_i = 2
            
            # Then check l=2, 3, ...
            
            # We can break if dp_j[l] > gap.
            
            # Let's try to limit l to best_len + 1?
            # Actually, we can limit l to the max l such that dp_j[l] < INF.
            # But we don't store that explicitly.
            
            # Let's just iterate up to best_len + 1.
            # If best_len is 20, we do 20 iterations.
            
            for l in range(1, best_len + 1):
                if dp_j[l] > gap:
                    break
                new_len = l + 1
                if gap < dp_i[new_len]:
                    dp_i[new_len] = gap
                    par_i[new_len] = j
                    if new_len > max_len_at_i:
                        max_len_at_i = new_len
                        
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
