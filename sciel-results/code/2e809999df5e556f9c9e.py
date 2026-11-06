import time
import math
import bisect

def construct():
    t0 = time.time()
    DEADLINE = 85.0  # safety margin

    N_LIMIT = 10000
    
    # Sieve of Eratosthenes
    is_prime = bytearray(b'\x01') * (N_LIMIT + 1)
    is_prime[0] = 0
    is_prime[1] = 0
    for p in range(2, int(math.isqrt(N_LIMIT)) + 1):
        if is_prime[p]:
            is_prime[p*p::p] = bytearray(len(is_prime[p*p::p]))
            
    primes = [i for i, val in enumerate(is_prime) if val]
    n = len(primes)
    
    if n < 2:
        return primes
        
    # dp[i][j] = length of longest valid subsequence ending with primes[j], primes[i]
    # where j < i
    # We store prev[i][j] to reconstruct the path
    dp = [[0] * n for _ in range(n)]
    prev = [[-1] * n for _ in range(n)]
    
    max_len = 0
    best_i = -1
    best_j = -1
    
    # Precompute suffix max and argmax for each row j to allow O(1) RMQ queries
    # suf_val[j][k] = max(dp[j][k], dp[j][k+1], ..., dp[j][j-1])
    # suf_idx[j][k] = argmax index in that range
    
    # We can compute these on the fly or precompute all.
    # Precomputing all suffix tables:
    # For each j, we need suf_val[j] and suf_idx[j] of size j.
    # Total memory: sum(j for j in range(n)) ~ n^2/2 ~ 750k integers. Fine.
    # Total time: sum(j for j in range(n)) ~ n^2/2 ~ 750k ops. Fine.
    
    # However, dp[j][k] is computed when we process column j?
    # No, dp[i][j] is computed when we process column j.
    # But dp[j][k] for k < j is already computed in previous iterations (when column k was processed? No).
    # Let's trace:
    # dp[i][j] depends on dp[j][k].
    # dp[j][k] is the length ending at primes[j] with previous primes[k].
    # This is computed when we process column k?
    # In the loop `for j in range(n)`, we compute dp[i][j] for all i > j.
    # So when we are at iteration `j`, we have just finished computing all dp[?][j]?
    # No.
    # Iteration 0: computes dp[i][0] for i > 0.
    # Iteration 1: computes dp[i][1] for i > 1. Needs dp[1][0]. dp[1][0] was computed in Iteration 0.
    # Iteration j: computes dp[i][j] for i > j. Needs dp[j][k] for k < j.
    # dp[j][k] was computed in Iteration k (where i=j was one of the values).
    # So by the time we start Iteration j, all dp[j][k] for k < j are already known.
    # Therefore, we can build the suffix table for row j at the start of Iteration j.
    
    for j in range(n):
        if j == 0:
            for i in range(1, n):
                dp[i][0] = 2
                prev[i][0] = -1
                if dp[i][0] > max_len:
                    max_len = dp[i][0]
                    best_i = i
                    best_j = 0
        else:
            # Build suffix max and argmax for row j, indices 0 to j-1
            suf_val = [0] * j
            suf_idx = [-1] * j
            
            # Initialize last element
            suf_val[j-1] = dp[j][j-1]
            suf_idx[j-1] = j-1
            
            # Fill backwards
            for k in range(j-2, -1, -1):
                if dp[j][k] > suf_val[k+1]:
                    suf_val[k] = dp[j][k]
                    suf_idx[k] = k
                else:
                    # Tie-breaking: keep smaller index?
                    # The problem doesn't specify, but smaller index might be better for future extensions?
                    # Actually, any valid predecessor works.
                    # However, if we want to maximize length, the value is the same.
                    # Let's keep the one with smaller index to be consistent or just keep k+1's index?
                    # If we keep k+1's index, it's the "later" prime.
                    # If we keep k, it's the "earlier" prime.
                    # Does it matter?
                    # dp[i][j] = 1 + max_val.
                    # prev[i][j] = argmax.
                    # The sequence is reconstructed from prev.
                    # The constraint for the NEXT step (if we extend from i) depends on gap primes[i]-primes[j].
                    # The choice of k affects the gap primes[j]-primes[k].
                    # But dp[j][k] already encodes a valid sequence ending in k, j.
                    # So any k achieving the max is valid.
                    # Let's just pick the first one encountered in the backward scan (which is the smaller index k).
                    # Wait, the loop goes from j-2 down to 0.
                    # If dp[j][k] == suf_val[k+1], we currently assign suf_idx[k] = suf_idx[k+1] (implicitly by not updating? No, we need to set it).
                    # In the code above:
                    # if dp[j][k] > suf_val[k+1]: ...
                    # else: suf_val[k] = suf_val[k+1]; suf_idx[k] = suf_idx[k+1]
                    # This picks the index from the right (larger index).
                    # Let's stick to that for consistency.
                    suf_val[k] = suf_val[k+1]
                    suf_idx[k] = suf_idx[k+1]
            
            # Now compute dp[i][j] for all i > j
            for i in range(j + 1, n):
                gap = primes[i] - primes[j]
                target = primes[j] - gap
                # Binary search in primes[0..j-1] for smallest k with primes[k] >= target
                # bisect_left(primes, target, 0, j)
                k_min = bisect.bisect_left(primes, target, 0, j)
                
                if k_min < j:
                    max_val = suf_val[k_min]
                    dp[i][j] = 1 + max_val
                    prev[i][j] = suf_idx[k_min]
                else:
                    dp[i][j] = 2
                    prev[i][j] = -1
                    
                if dp[i][j] > max_len:
                    max_len = dp[i][j]
                    best_i = i
                    best_j = j
                    
        if time.time() - t0 > DEADLINE:
            break

    if best_i == -1:
        return primes[:1]
        
    seq = [primes[best_i], primes[best_j]]
    cur_i = best_i
    cur_j = best_j
    while prev[cur_i][cur_j] != -1:
        cur_k = prev[cur_i][cur_j]
        seq.append(primes[cur_k])
        cur_i, cur_j = cur_j, cur_k
        
    seq.reverse()
    return seq
