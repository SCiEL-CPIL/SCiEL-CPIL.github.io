import time
import math
import numpy as np

def construct():
    t0 = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
    N_LIMIT = 10000
    
    # Sieve of Eratosthenes using bytearray for speed
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
    # where j < i. We store prev[i][j] to reconstruct the path.
    # n is ~1229, so n^2 is ~1.5M. Memory is fine.
    dp = [[0] * n for _ in range(n)]
    prev = [[-1] * n for _ in range(n)]
    
    max_len = 0
    best_i = -1
    best_j = -1
    
    # Precompute logs for sparse table
    max_log = int(math.log2(n)) + 1
    logs = [0] * (n + 1)
    for i in range(2, n + 1):
        logs[i] = logs[i // 2] + 1
        
    # We process j from 0 to n-1
    # For each j, we compute dp[i][j] for all i > j
    # dp[i][j] = 1 + max(dp[k][j] for k in [k_min, j-1]) where primes[k] >= 2*primes[j] - primes[i]
    # If no valid k, dp[i][j] = 2
    
    for j in range(n):
        if time.time() - t0 > DEADLINE:
            break
            
        if j == 0:
            # No previous elements, so any pair starting with primes[0] has length 2
            for i in range(1, n):
                dp[i][0] = 2
                prev[i][0] = -1
                if 2 > max_len:
                    max_len = 2
                    best_i = i
                    best_j = 0
        else:
            # Build sparse table for row j: st[p][k] = max(dp[j][k] for k in [k, k+2^p-1])
            size = j
            K = logs[size] + 1
            st = [[0] * size for _ in range(K)]
            
            # Base case
            for k in range(size):
                st[0][k] = dp[j][k]
                
            for p in range(1, K):
                span = 1 << p
                half = 1 << (p - 1)
                limit = size - span + 1
                for k in range(limit):
                    st[p][k] = max(st[p-1][k], st[p-1][k + half])
            
            # Helper for RMQ on row j
            def rmq(l, r):
                if l > r:
                    return 0
                length = r - l + 1
                k = logs[length]
                return max(st[k][l], st[k][r - (1 << k) + 1])
            
            # For each i > j, find max dp[k][j] for valid k
            # Use monotonic pointer: as i increases, threshold decreases, so k_min moves left
            k_ptr = j - 1
            current_max = 0
            best_k = -1
            
            for i in range(j + 1, n):
                if time.time() - t0 > DEADLINE:
                    break
                    
                threshold = 2 * primes[j] - primes[i]
                
                # Move k_ptr left while primes[k_ptr] >= threshold
                while k_ptr >= 0 and primes[k_ptr] >= threshold:
                    if dp[j][k_ptr] > current_max:
                        current_max = dp[j][k_ptr]
                        best_k = k_ptr
                    k_ptr -= 1
                
                if current_max > 0:
                    new_len = current_max + 1
                    dp[i][j] = new_len
                    prev[i][j] = best_k
                    if new_len > max_len:
                        max_len = new_len
                        best_i = i
                        best_j = j
                else:
                    dp[i][j] = 2
                    prev[i][j] = -1
                    if 2 > max_len:
                        max_len = 2
                        best_i = i
                        best_j = j
    
    # Reconstruct sequence
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
