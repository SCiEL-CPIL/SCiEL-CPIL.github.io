import numpy as np
import time

def construct():
    start_time = time.time()
    
    limit = 10000
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(np.sqrt(limit)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    primes = np.array([i for i, val in enumerate(is_prime) if val], dtype=np.int32)
    N = len(primes)
    
    max_gap = limit
    
    dp = np.zeros((N, max_gap + 1), dtype=np.int16)
    best = np.zeros((N, max_gap + 1), dtype=np.int16)
    prev = np.full((N, max_gap + 1), -1, dtype=np.int16)
    
    for i in range(N):
        p_i = primes[i]
        for j in range(i):
            d = p_i - primes[j]
            val = 1 + best[j][d]
            if val > dp[i][d]:
                dp[i][d] = val
                prev[i][d] = j
                
        # Compute best[i]
        best[i] = np.maximum.accumulate(dp[i])
        
        if time.time() - start_time > 50:
            break

    # Find global max
    max_len = 0
    best_i = -1
    best_d = -1
    # We only need to check up to the index where we stopped if we broke early
    # But best[i][max_gap] is the max for i
    # If we broke, indices after are 0.
    for i in range(N):
        if best[i][max_gap] > max_len:
            max_len = best[i][max_gap]
            best_i = i
            # We don't know best_d yet, need to find it
            # best[i][max_gap] is the max.
            # We need the d that achieved it.
            # It's the argmax of dp[i][0:max_gap+1]
            # But we can just find it during backtracking or now.
            # Let's find it now.
            # Since best[i] is prefix max, the max is at max_gap.
            # The d that achieves it is the first d where dp[i][d] == best[i][max_gap]?
            # Not necessarily the first, but any.
            # We can use argmax on dp[i].
            # But we need to be consistent with backtracking.
            # In backtracking, we will compute d' using argmax.
            # So for the start, we can just pick the d that gives the max.
            # Let's pick the smallest such d.
            best_d = np.argmax(dp[i])
            
    # Backtrack
    seq = []
    cur_i = best_i
    cur_d = best_d
    
    while cur_i != -1:
        seq.append(primes[cur_i])
        if cur_d == 0:
            break
        
        next_i = prev[cur_i][cur_d]
        # Find the gap d' in next_i that was used
        # It must satisfy dp[next_i][d'] == best[next_i][cur_d]
        # And d' <= cur_d
        # We use argmax on dp[next_i][0:cur_d+1]
        d_slice = dp[next_i][0:cur_d+1]
        cur_d = np.argmax(d_slice)
        cur_i = next_i
        
    seq.reverse()
    return seq
