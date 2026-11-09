import bisect
import time

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    # 1. Sieve of Eratosthenes
    limit = 10000
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, val in enumerate(is_prime) if val]
    n = len(primes)
    
    # 2. DP Setup
    # L[i][j] = max length of valid subsequence ending with pair (primes[j], primes[i])
    # prev[i][j] = k such that L[i][j] = L[j][k] + 1
    L = [[0] * n for _ in range(n)]
    prev = [[-1] * n for _ in range(n)]
    
    # We process j from 0 to n-1
    # For a fixed j, we need to answer queries for all i > j
    # The queries depend on L[j][k] for k < j
    # So we build a structure for L[j][0..j-1]
    
    for j in range(n):
        if time.time() > deadline:
            break
            
        if j == 0:
            # No k < 0, so L[i][0] will be 2 for all i > 0
            suf_val = None
            suf_idx = None
        else:
            # Build suffix max arrays for L[j][0..j-1]
            # suf_val[k] = max(L[j][k], L[j][k+1], ..., L[j][j-1])
            # suf_idx[k] = argmax index
            suf_val = [0] * j
            suf_idx = [-1] * j
            for k in range(j - 1, -1, -1):
                if k == j - 1:
                    suf_val[k] = L[j][k]
                    suf_idx[k] = k
                else:
                    if L[j][k] >= suf_val[k + 1]:
                        suf_val[k] = L[j][k]
                        suf_idx[k] = k
                    else:
                        suf_val[k] = suf_val[k + 1]
                        suf_idx[k] = suf_idx[k + 1]

        # Iterate i > j
        for i in range(j + 1, n):
            if time.time() > deadline:
                break
                
            gap_ij = primes[i] - primes[j]
            # We need k < j such that primes[j] - primes[k] <= gap_ij
            # primes[k] >= primes[j] - gap_ij
            target = primes[j] - gap_ij
            
            # Find smallest k in [0, j-1] such that primes[k] >= target
            k_min = bisect.bisect_left(primes, target, 0, j)
            
            if k_min < j:
                best_val = suf_val[k_min]
                best_k = suf_idx[k_min]
                if best_val > 0:
                    L[i][j] = best_val + 1
                    prev[i][j] = best_k
                else:
                    L[i][j] = 2
                    prev[i][j] = -1
            else:
                L[i][j] = 2
                prev[i][j] = -1

    # Find the best end state
    max_len = 0
    best_i = -1
    best_j = -1
    for i in range(n):
        for j in range(i):
            if L[i][j] > max_len:
                max_len = L[i][j]
                best_i = i
                best_j = j
                
    # If no pair found (n < 2), return primes[:1] or similar
    if max_len == 0:
        return primes[:1] if n > 0 else []
        
    # Backtrack
    seq_indices = []
    cur_i = best_i
    cur_j = best_j
    while cur_j != -1:
        seq_indices.append(cur_i)
        next_k = prev[cur_i][cur_j]
        cur_i = cur_j
        cur_j = next_k
    seq_indices.append(cur_i)
    
    seq_indices.reverse()
    result = [primes[idx] for idx in seq_indices]
    
    return result
