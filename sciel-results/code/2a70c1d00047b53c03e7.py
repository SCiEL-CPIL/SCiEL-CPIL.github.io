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
    # D[i][j] = max length of sequence ending with primes[i], primes[j] (i < j)
    # We store D as a list of lists.
    D = [[0] * n_primes for _ in range(n_primes)]
    prev = [[-1] * n_primes for _ in range(n_primes)]
    
    best_len = 2
    best_i = 0
    best_j = 1
    
    # Iterate i as the second-to-last prime index
    for i in range(n_primes - 1):
        # Build suffix max value and index arrays for column i (rows 0 to i-1)
        # M[k] = max(D[t][i] for t in [k, i-1])
        # M_idx[k] = argmax(D[t][i] for t in [k, i-1])
        
        if i > 0:
            M_val = [0] * i
            M_idx = [-1] * i
            
            M_val[i-1] = D[i-1][i]
            M_idx[i-1] = i-1
            
            for k in range(i-2, -1, -1):
                if D[k][i] > M_val[k+1]:
                    M_val[k] = D[k][i]
                    M_idx[k] = k
                else:
                    M_val[k] = M_val[k+1]
                    M_idx[k] = M_idx[k+1]
        else:
            M_val = []
            M_idx = []
            
        # Two-pointer optimization for k_min
        # k_min is the smallest k < i such that primes[k] >= threshold
        # As j increases, threshold decreases, so k_min is non-increasing.
        
        # Initialize k_min for the first j (i+1)
        # If i == 0, there are no k < 0, so skip inner loop? 
        # The loop for j is range(i+1, n_primes).
        # If i=0, k_min would be invalid. 
        # In the original code, if k_min >= i, it's a base case.
        
        # We need to handle the case where no valid k exists.
        # Let's initialize k_min to i (meaning "none found yet" or "out of bounds")
        # But we need the *smallest* k.
        # Let's find the initial k_min using bisect for j = i+1.
        
        if i > 0:
            # Find initial k_min for j = i+1
            gap_init = primes[i+1] - primes[i]
            threshold_init = primes[i] - gap_init
            k_min = bisect.bisect_left(primes, threshold_init, 0, i)
            
            for j in range(i + 1, n_primes):
                gap = primes[j] - primes[i]
                threshold = primes[i] - gap
                
                # Update k_min if necessary
                # k_min is the smallest index with primes[k] >= threshold
                # We know k_min from previous step was valid for higher threshold.
                # New threshold is lower.
                # So primes[k_min - 1] might now be >= threshold.
                while k_min > 0 and primes[k_min - 1] >= threshold:
                    k_min -= 1
                
                # Now k_min is the smallest index such that primes[k_min] >= threshold
                # Check if k_min < i
                if k_min < i:
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
                    if 2 > best_len:
                        best_len = 2
                        best_i = i
                        best_j = j
        else:
            # i == 0, no k < 0
            for j in range(i + 1, n_primes):
                D[i][j] = 2
                prev[i][j] = -1
                if 2 > best_len:
                    best_len = 2
                    best_i = i
                    best_j = j

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
