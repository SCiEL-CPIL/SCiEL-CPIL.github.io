import time
from math import isqrt
import bisect

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    N_LIMIT = 10000
    
    # 1. Sieve of Eratosthenes
    is_prime = [True] * (N_LIMIT + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, isqrt(N_LIMIT) + 1):
        if is_prime[i]:
            for j in range(i*i, N_LIMIT + 1, i):
                is_prime[j] = False
                
    primes = [i for i, v in enumerate(is_prime) if v]
    n_primes = len(primes)
    
    if n_primes < 2:
        return primes
        
    # 2. DP Setup
    # D[i][j] = max length of sequence ending with primes[i], primes[j]
    # We use lists of lists for speed and simplicity
    # n_primes is ~1229, so 1229x1229 matrix is ~1.5M elements.
    # Python ints are objects, so memory is ~1.5M * 28 bytes ~ 42MB. Acceptable.
    
    D = [[0] * n_primes for _ in range(n_primes)]
    prev = [[-1] * n_primes for _ in range(n_primes)]
    
    best_len = 2
    best_i = 0
    best_j = 1
    
    # We iterate i from 0 to n_primes - 2
    # For each i, we compute suffix max arrays for column i (rows 0..i-1)
    # Then we update D[i][j] for all j > i
    
    # Pre-allocate suf_val and suf_idx for current i
    # We only need one set at a time, so we can reuse lists or create new ones.
    # Creating new lists of size i for each i is O(N^2) total allocation, which is fine.
    
    for i in range(n_primes - 1):
        
        # Build suffix max value and index arrays for column i (rows 0 to i-1)
        # M_val[k] = max(D[t][i] for t in [k, i-1])
        # M_idx[k] = argmax(D[t][i] for t in [k, i-1])
        
        if i > 0:
            # M arrays of length i
            M_val = [0] * i
            M_idx = [-1] * i
            
            # Initialize with last element
            M_val[i-1] = D[i-1][i]
            M_idx[i-1] = i-1
            
            # Fill backwards
            for k in range(i-2, -1, -1):
                if D[k][i] >= M_val[k+1]:
                    M_val[k] = D[k][i]
                    M_idx[k] = k
                else:
                    M_val[k] = M_val[k+1]
                    M_idx[k] = M_idx[k+1]
        else:
            M_val = []
            M_idx = []
            
        # Now update D[i][j] for j > i
        for j in range(i + 1, n_primes):
            gap = primes[j] - primes[i]
            # Condition: primes[i] - primes[k] <= gap
            # primes[k] >= primes[i] - gap
            threshold = primes[i] - gap
            
            # Find smallest k < i such that primes[k] >= threshold
            # Search in primes[0:i]
            k_min = bisect.bisect_left(primes, threshold, 0, i)
            
            if k_min < i:
                # Best predecessor is in [k_min, i-1]
                best_prev_len = M_val[k_min]
                best_k = M_idx[k_min]
                
                new_len = best_prev_len + 1
                D[i][j] = new_len
                prev[i][j] = best_k
                
                if new_len > best_len:
                    best_len = new_len
                    best_i = i
                    best_j = j
            else:
                # No valid predecessor k. Base case: sequence is just P[i], P[j]
                D[i][j] = 2
                prev[i][j] = -1
                # best_len is at least 2, so no update needed unless best_len < 2
                # but we initialized best_len = 2
                
        # Deadline check
        if time.time() > deadline:
            break
            
    # Reconstruct sequence
    if best_len < 2:
        return primes[:best_len]
        
    seq = [primes[best_j], primes[best_i]]
    cur_i = best_i
    cur_j = best_j
    
    while prev[cur_i][cur_j] != -1:
        k = prev[cur_i][cur_j]
        seq.append(primes[k])
        cur_j = cur_i
        cur_i = k
        
    seq.reverse()
    
    return seq
