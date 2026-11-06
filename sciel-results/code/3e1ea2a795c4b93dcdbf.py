"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from bisect import bisect_left

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    # 1. Generate primes <= 10000
    limit = 10000
    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False

    primes = [i for i, flag in enumerate(is_prime) if flag]
    N = len(primes)

    if N < 2:
        return primes

    # dp[i][j] = max length of valid sequence ending with primes[i], primes[j] (i < j)
    # We store dp as a list of arrays for each j, where dp[j] is a numpy-like list of length j
    # Actually, let's use standard lists but be smarter about memory layout.
    # The current code uses dp = [[0] * N for _ in range(N)] which is N^2 integers.
    # 1229^2 ~ 1.5M integers, which is fine.
    
    # However, the suffix max arrays suf_max_val and suf_max_arg take O(N^2) space as well.
    # Let's try to optimize the inner loop.
    
    # Key insight: For a fixed j, we compute dp[i][j] for all i < j.
    # dp[i][j] depends on suf_max_val[i][L] where L = bisect_left(primes, primes[i] - (primes[j]-primes[i]), 0, i)
    # This L depends on both i and j.
    
    # Alternative formulation:
    # Let f(j) = max length of valid sequence ending at prime j.
    # But we need the last gap to be non-decreasing, so we need to know the previous prime.
    # So dp[i][j] is necessary.
    
    # Let's try a different optimization:
    # For each j, instead of computing all dp[i][j] from scratch, can we reuse information?
    # The suffix max arrays are already an optimization.
    
    # Let me try to vectorize some of the operations or use a more efficient data structure.
    
    # Actually, let's reconsider the problem. The gaps are non-decreasing.
    # If we have a sequence p_{a1} < p_{a2} < ... < p_{ak}, then
    # p_{a2} - p_{a1} <= p_{a3} - p_{a2} <= ... <= p_{ak} - p_{a_{k-1}}
    
    # This is equivalent to: for each consecutive triple, the middle prime is at most the average
    # of its neighbors? No, that's for convex sequences. For non-decreasing gaps, we have
    # p_{a_{i+1}} - p_{a_i} >= p_{a_i} - p_{a_{i-1}}, which means
    # p_{a_{i+1}} + p_{a_{i-1}} >= 2 * p_{a_i}.
    # So the sequence is "convex" in the sense that each term is at most the average of its neighbors.
    
    # This might allow a greedy approach? Let's think...
    # If we greedily choose the next prime to be as small as possible while maintaining non-decreasing gaps,
    # would that maximize the length? Not necessarily, because choosing a slightly larger prime now
    # might allow more primes later.
    
    # Let's stick with the DP but try to make it faster.
    
    # One optimization: use numpy for the suffix max computations.
    import numpy as np
    
    dp = [None] * N
    prev = [None] * N
    dp[0] = [0]  # placeholder
    prev[0] = [-1]
    
    suf_max_val = [None] * N
    suf_max_arg = [None] * N
    
    max_len = 1
    best_i = 0
    best_j = 0
    
    p = primes
    bl = bisect_left
    time_check = time.time
    
    for j in range(1, N):
        if time_check() > deadline:
            break
        
        pj = p[j]
        dp_j = [0] * j
        prev_j = [-1] * j
        
        for i in range(0, j):
            gap = pj - p[i]
            threshold = p[i] - gap
            
            L = bl(p, threshold, 0, i)
            
            if L < i:
                val = suf_max_val[i][L]
                if val > 0:
                    dp_j[i] = val + 1
                    prev_j[i] = suf_max_arg[i][L]
                else:
                    dp_j[i] = 2
            else:
                dp_j[i] = 2
            
            if dp_j[i] > max_len:
                max_len = dp_j[i]
                best_i = i
                best_j = j
        
        dp[j] = dp_j
        prev[j] = prev_j
        
        # Compute suffix max for column j
        if j > 0:
            sm_val = [0] * j
            sm_arg = [-1] * j
            sm_val[j-1] = dp_j[j-1]
            sm_arg[j-1] = j-1
            for k in range(j-2, -1, -1):
                if dp_j[k] > sm_val[k+1]:
                    sm_val[k] = dp_j[k]
                    sm_arg[k] = k
                else:
                    sm_val[k] = sm_val[k+1]
                    sm_arg[k] = sm_arg[k+1]
            suf_max_val[j] = sm_val
            suf_max_arg[j] = sm_arg
    
    # Reconstruct sequence
    seq = [primes[best_j]]
    curr_i = best_i
    curr_j = best_j
    while curr_i != -1:
        seq.append(primes[curr_i])
        prev_i = prev[curr_j][curr_i]
        curr_j = curr_i
        curr_i = prev_i
    
    seq.reverse()
    
    return seq

# EVOLVE-BLOCK-END
