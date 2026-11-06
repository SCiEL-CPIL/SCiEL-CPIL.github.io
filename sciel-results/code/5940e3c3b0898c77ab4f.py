import sys
import time
import bisect

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin
    
    # Sieve of Eratosthenes for primes up to 10000
    limit = 10000
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
            
    primes = [i for i, prime in enumerate(is_prime) if prime]
    N = len(primes)
    
    if N == 0:
        return []
    if N == 1:
        return primes
    
    # dp[i][j] = length of longest valid subsequence ending with primes[i], primes[j]
    # prev[i][j] = index k such that subsequence ends with primes[k], primes[i], primes[j]
    # We only need to store dp and prev for i < j
    
    # To save memory and time, we can use lists of lists.
    # dp[i] is a list of length N, but we only fill indices > i.
    # Actually, we can just use a 2D array of size N x N.
    
    dp = [[0] * N for _ in range(N)]
    prev = [[-1] * N for _ in range(N)]
    
    max_len = 2
    best_i = 0
    best_j = 1
    
    # Iterate i from 0 to N-1
    for i in range(N):
        # Check time
        if time.time() > deadline:
            # Return best found so far
            # Reconstruct
            break
            
        # Prepare column i data: dp[k][i] for k in 0..i-1
        # We need SufMax of these values.
        # SufMax[x] = (max_value, index) for range [x, i-1]
        
        if i == 0:
            # No k < 0
            # dp[0][j] = 2 for all j > 0
            for j in range(1, N):
                dp[0][j] = 2
                if 2 > max_len:
                    max_len = 2
                    best_i = 0
                    best_j = j
        else:
            # Build SufMax for column i
            # Values are dp[0][i], dp[1][i], ..., dp[i-1][i]
            suf_max_val = [0] * i
            suf_max_idx = [0] * i
            
            # Initialize last element
            suf_max_val[i-1] = dp[i-1][i]
            suf_max_idx[i-1] = i-1
            
            for x in range(i-2, -1, -1):
                if dp[x][i] > suf_max_val[x+1]:
                    suf_max_val[x] = dp[x][i]
                    suf_max_idx[x] = x
                else:
                    suf_max_val[x] = suf_max_val[x+1]
                    suf_max_idx[x] = suf_max_idx[x+1]
            
            # Find initial k_ptr for j = i+1
            # target = 2 * primes[i] - primes[i+1]
            # k_ptr = bisect_left(primes, target, 0, i)
            
            target = 2 * primes[i] - primes[i+1]
            k_ptr = bisect.bisect_left(primes, target, 0, i)
            
            for j in range(i+1, N):
                if j > i + 1:
                    target = 2 * primes[i] - primes[j]
                    # Update k_ptr
                    # k_ptr is the smallest index such that primes[k] >= previous_target
                    # New target is smaller, so we might move k_ptr left
                    while k_ptr > 0 and primes[k_ptr - 1] >= target:
                        k_ptr -= 1
                
                if k_ptr < i:
                    val = 1 + suf_max_val[k_ptr]
                    idx = suf_max_idx[k_ptr]
                else:
                    val = 2
                    idx = -1
                
                dp[i][j] = val
                prev[i][j] = idx
                
                if val > max_len:
                    max_len = val
                    best_i = i
                    best_j = j
    
    # Reconstruct sequence
    if max_len == 1:
        return primes[:1]
        
    seq = []
    cur_i = best_i
    cur_j = best_j
    seq.append(primes[cur_j])
    seq.append(primes[cur_i])
    
    while prev[cur_i][cur_j] != -1:
        k = prev[cur_i][cur_j]
        seq.append(primes[k])
        cur_j = cur_i
        cur_i = k
        
    seq.reverse()
    return seq

if __name__ == "__main__":
    print(construct())
