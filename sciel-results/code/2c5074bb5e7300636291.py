import time
import bisect

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin before 60s limit
    
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
    # We use a list of lists.
    dp = [[0] * N for _ in range(N)]
    prev = [[-1] * N for _ in range(N)]
    
    # suf_max_val[i] and suf_max_idx[i] for column i
    # suf_max_val[i][k] = max(dp[m][i]) for m in [k, i-1]
    # suf_max_idx[i][k] = argmax m in [k, i-1]
    # If i=0, empty. If i=1, size 1 (k=0).
    suf_max_val = [None] * N
    suf_max_idx = [None] * N
    
    max_len = 1
    best_i = 0
    best_j = 0
    
    # Base cases: sequences of length 1 are trivial.
    # Our dp handles length >= 2.
    
    for j in range(1, N):
        # Check time
        if time.time() > deadline:
            break
            
        # Compute dp[i][j] for all i < j
        for i in range(0, j):
            gap = primes[j] - primes[i]
            threshold = primes[i] - gap
            
            # Find smallest k < i such that primes[k] >= threshold
            # primes[0:i] is sorted
            # bisect_left returns the first index >= threshold
            L = bisect.bisect_left(primes, threshold, 0, i)
            
            if L < i:
                # We have valid k in [L, i-1]
                # Get max from suf_max_val[i]
                val = suf_max_val[i][L]
                idx = suf_max_idx[i][L]
                
                dp[i][j] = val + 1
                prev[i][j] = idx
            else:
                dp[i][j] = 2
                prev[i][j] = -1
            
            if dp[i][j] > max_len:
                max_len = dp[i][j]
                best_i = i
                best_j = j
        
        # Compute suf_max for column j
        # Column j has entries dp[0][j], ..., dp[j-1][j]
        # We need suf_max_val[j] of size j
        # suf_max_val[j][k] = max(dp[m][j]) for m in [k, j-1]
        if j > 0:
            sm_val = [0] * j
            sm_idx = [0] * j
            
            # Initialize with last element m = j-1
            sm_val[j-1] = dp[j-1][j]
            sm_idx[j-1] = j-1
            
            for k in range(j-2, -1, -1):
                if dp[k][j] > sm_val[k+1]:
                    sm_val[k] = dp[k][j]
                    sm_idx[k] = k
                else:
                    sm_val[k] = sm_val[k+1]
                    sm_idx[k] = sm_idx[k+1]
            
            suf_max_val[j] = sm_val
            suf_max_idx[j] = sm_idx
    
    # Reconstruct sequence
    # Start from best_j, best_i
    # Sequence ends with primes[best_i], primes[best_j]
    # prev[best_i][best_j] is the index k such that the sequence ended with primes[k], primes[best_i]
    
    seq = [primes[best_j]]
    curr_i = best_i
    curr_j = best_j
    
    # Safety check to prevent infinite loops or errors
    steps = 0
    while curr_i != -1 and steps < N:
        seq.append(primes[curr_i])
        prev_i = prev[curr_i][curr_j]
        curr_j = curr_i
        curr_i = prev_i
        steps += 1
    
    seq.reverse()
    
    # Verify result (optional, for debugging)
    # if len(seq) > 1:
    #    gaps = [seq[i+1]-seq[i] for i in range(len(seq)-1)]
    #    for i in range(len(gaps)-1):
    #        if gaps[i] > gaps[i+1]:
    #            print("Invalid gaps", gaps)
    #            break
    
    return seq
