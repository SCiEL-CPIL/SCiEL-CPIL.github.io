"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import bisect
import time
from math import isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin for safety

    # Generate primes up to 10000
    primes = []
    for p in range(2, 10001):
        if all(p % d for d in range(2, isqrt(p) + 1)):
            primes.append(p)
            
    N = len(primes)
    if N == 0:
        return []
    if N == 1:
        return primes
    
    # dp[i][j] = length of longest sequence ending with primes[i], primes[j] (i < j)
    # parent[i][j] = index k such that dp[i][j] = dp[k][i] + 1
    dp = [[0] * N for _ in range(N)]
    parent = [[-1] * N for _ in range(N)]
    
    # sm_val[i] is a list of length i, where sm_val[i][x] = max(dp[k][i] for k in [x, i-1])
    # sm_idx[i] is a list of length i, where sm_idx[i][x] = argmax(dp[k][i] for k in [x, i-1])
    sm_val = [None] * N
    sm_idx = [None] * N
    
    max_len = 2
    best_i = 0
    best_j = 1
    
    # Iterate j (last prime index) from 1 to N-1
    for j in range(1, N):
        if time.time() > deadline:
            # Return best found so far
            break
        
        for i in range(0, j):
            # Condition: primes[j] - primes[i] >= primes[i] - primes[k]
            # => primes[k] >= 2*primes[i] - primes[j]
            L = 2 * primes[i] - primes[j]
            
            # Find smallest k < i such that primes[k] >= L
            # Search in primes[0...i-1]
            k_min = bisect.bisect_left(primes, L, 0, i)
            
            if k_min < i:
                # There is at least one valid k
                # Get max dp[k][i] for k in [k_min, i-1]
                best_prev_len = sm_val[i][k_min]
                best_prev_idx = sm_idx[i][k_min]
                
                dp[i][j] = best_prev_len + 1
                parent[i][j] = best_prev_idx
                
                if dp[i][j] > max_len:
                    max_len = dp[i][j]
                    best_i = i
                    best_j = j
            else:
                # No valid k, sequence length is 2
                dp[i][j] = 2
                parent[i][j] = -1
                if 2 > max_len:
                    max_len = 2
                    best_i = i
                    best_j = j

        # Build sm_val[j] and sm_idx[j]
        # sm_val[j][x] = max(dp[k][j] for k in [x, j-1])
        sm_val[j] = [0] * j
        sm_idx[j] = [0] * j
        
        # Initialize with the last element
        sm_val[j][j-1] = dp[j-1][j]
        sm_idx[j][j-1] = j-1
        
        for x in range(j-2, -1, -1):
            if dp[x][j] > sm_val[j][x+1]:
                sm_val[j][x] = dp[x][j]
                sm_idx[j][x] = x
            else:
                sm_val[j][x] = sm_val[j][x+1]
                sm_idx[j][x] = sm_idx[j][x+1]
                
    # Reconstruct the sequence
    if max_len == 0:
        return primes[:1] # Should not happen if N>=1
        
    seq = [primes[best_j], primes[best_i]]
    i, j = best_i, best_j
    k = parent[i][j]
    while k != -1:
        seq.append(primes[k])
        j = i
        i = k
        k = parent[i][j]
        
    seq.reverse()
    return seq

# EVOLVE-BLOCK-END
