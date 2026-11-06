import time
import math
import bisect

def construct():
    # Deadline guard
    deadline = time.time() + 85
    
    # Sieve of Eratosthenes
    limit = 10000
    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i * i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, v in enumerate(is_prime) if v]
    N = len(primes)
    
    if N <= 1:
        return primes
        
    # dp[i][j] = max length of sequence ending with primes[i], primes[j]
    # parent[i][j] = index h such that sequence ending with primes[h], primes[i] is the predecessor
    # If parent[i][j] == -1, the sequence is just [primes[i], primes[j]]
    
    # We use lists of lists for dp and parent
    # dp[i][j] is only valid for i < j
    dp = [[0] * N for _ in range(N)]
    parent = [[-1] * N for _ in range(N)]
    
    max_len = 1
    max_i, max_j = 0, 0
    
    # Iterate i from 0 to N-2
    for i in range(N - 1):
        if time.time() > deadline:
            break
            
        # If i > 0, prepare suffix max data for column i (values dp[h][i] for h < i)
        # We need to query max dp[h][i] for h in [h_min, i-1]
        # Since h_min depends on j, we precompute suffix maxes for the array dp[0..i-1][i]
        
        if i > 0:
            # suffix_max_val[k] = max(dp[k][i], dp[k+1][i], ..., dp[i-1][i])
            # suffix_max_idx[k] = argmax index in [k, i-1]
            suffix_max_val = [0] * i
            suffix_max_idx = [0] * i
            
            # Initialize for the last element (i-1)
            suffix_max_val[i-1] = dp[i-1][i]
            suffix_max_idx[i-1] = i-1
            
            for k in range(i - 2, -1, -1):
                if dp[k][i] > suffix_max_val[k+1]:
                    suffix_max_val[k] = dp[k][i]
                    suffix_max_idx[k] = k
                else:
                    suffix_max_val[k] = suffix_max_val[k+1]
                    suffix_max_idx[k] = suffix_max_idx[k+1]
            
            # Now iterate j from i+1 to N-1
            for j in range(i + 1, N):
                if time.time() > deadline:
                    break
                    
                gap_j = primes[j] - primes[i]
                # Condition for h: primes[i] - primes[h] <= gap_j
                # primes[h] >= primes[i] - gap_j
                threshold = primes[i] - gap_j
                
                # Find smallest index h_min in [0, i-1] such that primes[h_min] >= threshold
                # primes[0..i-1] is sorted
                # bisect_left finds the first index where value >= threshold
                h_min = bisect.bisect_left(primes, threshold, 0, i)
                
                if h_min < i:
                    # There is at least one valid h
                    prev_len = suffix_max_val[h_min]
                    prev_idx = suffix_max_idx[h_min]
                    
                    dp[i][j] = prev_len + 1
                    parent[i][j] = prev_idx
                    
                    if dp[i][j] > max_len:
                        max_len = dp[i][j]
                        max_i = i
                        max_j = j
                else:
                    # No valid predecessor, sequence is just primes[i], primes[j]
                    dp[i][j] = 2
                    parent[i][j] = -1
                    
                    if dp[i][j] > max_len:
                        max_len = dp[i][j]
                        max_i = i
                        max_j = j
        else:
            # i == 0, no h < 0
            for j in range(1, N):
                if time.time() > deadline:
                    break
                dp[0][j] = 2
                parent[0][j] = -1
                if dp[0][j] > max_len:
                    max_len = dp[0][j]
                    max_i = 0
                    max_j = j

    # Reconstruct the sequence
    if max_len <= 1:
        return [primes[0]]
        
    seq = []
    i, j = max_i, max_j
    while True:
        seq.append(primes[j])
        h = parent[i][j]
        if h == -1:
            seq.append(primes[i])
            break
        j = i
        i = h
        
    seq.reverse()
    return seq

# EVOLVE-BLOCK-END
