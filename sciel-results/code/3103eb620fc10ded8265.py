import time
import bisect

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    
    # Generate primes <= 10000
    limit = 10000
    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    is_prime[1] = False
    for p in range(2, int(limit**0.5) + 1):
        if is_prime[p]:
            for multiple in range(p*p, limit + 1, p):
                is_prime[multiple] = False
    primes = [i for i, flag in enumerate(is_prime) if flag]
    N = len(primes)
    
    # dp[i][j] = max length of sequence ending in primes[i], primes[j]
    # prev[i][j] = index k such that dp[k][i] was the max used to compute dp[i][j]
    dp = [[0] * N for _ in range(N)]
    prev = [[-1] * N for _ in range(N)]
    
    # suf_max[j][k] = max(dp[k][j], ..., dp[j-1][j])
    # arg_suf[j][k] = index of the max in dp[k..j-1][j]
    suf_max = [None] * N
    arg_suf = [None] * N
    
    best_len = 0
    best_end = (-1, -1)
    
    for j in range(N):
        # Compute dp[i][j] for all i < j
        for i in range(j):
            # Base case: sequence of length 2
            curr_len = 2
            best_prev_idx = -1
            best_val = 0
            
            if i > 0:
                # Need max dp[k][i] for k < i such that primes[k] >= 2*primes[i] - primes[j]
                threshold = 2 * primes[i] - primes[j]
                # Find lower bound of threshold in primes
                L = bisect.bisect_left(primes, threshold)
                
                if L < i:
                    # Query suf_max[i][L]
                    val = suf_max[i][L]
                    if val > best_val:
                        best_val = val
                        best_prev_idx = arg_suf[i][L]
            
            if best_val > 0:
                curr_len = best_val + 1
            else:
                curr_len = 2
                best_prev_idx = -1
                
            dp[i][j] = curr_len
            prev[i][j] = best_prev_idx
            
            if curr_len > best_len:
                best_len = curr_len
                best_end = (i, j)
        
        # Build suf_max and arg_suf for column j
        # We need suf_max[j][k] for k in 0..j-1
        if j > 0:
            suf_max[j] = [0] * j
            arg_suf[j] = [0] * j
            
            # Initialize with last element
            suf_max[j][j-1] = dp[j-1][j]
            arg_suf[j][j-1] = j-1
            
            for k in range(j-2, -1, -1):
                if dp[k][j] > suf_max[j][k+1]:
                    suf_max[j][k] = dp[k][j]
                    arg_suf[j][k] = k
                else:
                    suf_max[j][k] = suf_max[j][k+1]
                    arg_suf[j][k] = arg_suf[j][k+1]
                    
        # Time check
        if time.time() > deadline:
            break
    
    # Reconstruct sequence
    if best_end[0] == -1:
        return []
        
    seq = []
    i, j = best_end
    while i != -1:
        seq.append(primes[j])
        seq.append(primes[i])
        j = i
        i = prev[i][j]
        
    seq.reverse()
    
    # Verify and return
    # The sequence should be increasing primes with non-decreasing gaps
    return seq
