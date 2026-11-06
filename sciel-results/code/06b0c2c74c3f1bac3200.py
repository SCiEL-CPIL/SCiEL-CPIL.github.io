import time
from math import isqrt

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
    # We use a list of lists. 
    # To save memory and improve cache locality, we can store D as a list of arrays or just lists.
    # D[i] is a list where D[i][j] is valid for j > i.
    # Actually, we only need D[k][i] for k < i when computing row i.
    # So we can store D as a 2D array D[i][j].
    
    D = [[0] * n_primes for _ in range(n_primes)]
    prev = [[-1] * n_primes for _ in range(n_primes)]
    
    # Initialize base cases: any pair (i, j) with i < j has length at least 2
    # But we will compute it in the loop.
    
    best_len = 2
    best_i = 0
    best_j = 1
    
    # Iterate i as the second-to-last prime index
    for i in range(n_primes - 1):
        # Pre-compute suffix maximums for D[k][i] for k < i
        # We need max(D[k][i]) for k in range [k_min, i-1]
        # Let's create an array col_vals where col_vals[k] = D[k][i]
        # And suffix_max[k] = max(col_vals[k], suffix_max[k+1])
        
        if i > 0:
            col_vals = [0] * i
            for k in range(i):
                col_vals[k] = D[k][i]
                
            suffix_max = [0] * (i + 1)
            # suffix_max[i] is unused effectively, or acts as 0
            suffix_max[i] = 0 
            for k in range(i - 1, -1, -1):
                suffix_max[k] = max(col_vals[k], suffix_max[k + 1])
        else:
            suffix_max = []
            
        # Iterate j as the last prime index
        for j in range(i + 1, n_primes):
            gap = primes[j] - primes[i]
            # Condition: primes[i] - primes[k] <= gap  => primes[k] >= primes[i] - gap
            threshold = primes[i] - gap
            
            # Find smallest k < i such that primes[k] >= threshold
            # Since primes are sorted, we can binary search in primes[0:i]
            
            # Binary search for threshold in primes[0:i]
            # We want the first index k such that primes[k] >= threshold
            lo, hi = 0, i
            while lo < hi:
                mid = (lo + hi) // 2
                if primes[mid] < threshold:
                    lo = mid + 1
                else:
                    hi = mid
                    
            k_min = lo
            
            if k_min < i:
                # The best k is in [k_min, i-1]
                # The max value is suffix_max[k_min]
                best_prev_len = suffix_max[k_min]
                
                # We also need to know WHICH k gave this max value to store in prev
                # suffix_max only stores the value. We need the index.
                # We can find the index by scanning backwards from i-1 to k_min? 
                # Or store indices in a parallel suffix array.
                
                # Let's refine: store (value, index) in suffix arrays?
                # Or just find the index that achieved the max.
                # Since we have col_vals, we can find the index.
                # But scanning takes O(i). Total O(N^3) worst case if we scan.
                # However, the range [k_min, i-1] might be small on average?
                # Actually, we can store the index of the max in the suffix array.
                
                # Let's re-prepare suffix arrays to store indices.
                # This requires rebuilding the logic inside the loop or pre-computing properly.
                
                # Optimization: Pre-compute suffix_max_val and suffix_max_idx for column i
                # We already did col_vals. Let's do it properly here.
                
                # To avoid O(i) scan for index, we should have stored indices.
                # Let's restart the inner logic with index tracking.
                
                # For efficiency, let's just do the binary search and then a linear scan 
                # to find the argmax in the valid range [k_min, i-1].
                # Is this too slow?
                # Worst case: i ~ N/2, range size ~ N/2.
                # Sum over i of (N-i) * (i) ~ N^3 / 6 ~ 10^9. Too slow.
                
                # We MUST store indices in the suffix max structure.
                
                pass # Placeholder, will implement below

    # Re-implementing the core loop with index tracking for suffix max
    
    # Clear D and prev to be safe if we re-run
    D = [[0] * n_primes for _ in range(n_primes)]
    prev = [[-1] * n_primes for _ in range(n_primes)]
    
    best_len = 2
    best_i = 0
    best_j = 1
    
    for i in range(n_primes - 1):
        # Build suffix max value and index arrays for column i (rows 0 to i-1)
        if i > 0:
            suffix_val = [0] * (i + 1)
            suffix_idx = [-1] * (i + 1)
            suffix_val[i] = 0
            suffix_idx[i] = -1
            
            for k in range(i - 1, -1, -1):
                val_k = D[k][i]
                if val_k >= suffix_val[k + 1]:
                    suffix_val[k] = val_k
                    suffix_idx[k] = k
                else:
                    suffix_val[k] = suffix_val[k + 1]
                    suffix_idx[k] = suffix_idx[k + 1]
        else:
            suffix_val = []
            suffix_idx = []
            
        for j in range(i + 1, n_primes):
            gap = primes[j] - primes[i]
            threshold = primes[i] - gap
            
            # Binary search for first k in [0, i-1] with primes[k] >= threshold
            lo, hi = 0, i
            while lo < hi:
                mid = (lo + hi) // 2
                if primes[mid] < threshold:
                    lo = mid + 1
                else:
                    hi = mid
                    
            k_min = lo
            
            if k_min < i:
                best_prev_len = suffix_val[k_min]
                best_k = suffix_idx[k_min]
                
                # The new length is best_prev_len + 1? 
                # Wait. D[k][i] is the length ending at k, i.
                # If we append j, the new sequence ends at i, j.
                # The length increases by 1?
                # D[k][i] = L. Sequence: ..., P[k], P[i].
                # New sequence: ..., P[k], P[i], P[j].
                # Length is L + 1.
                
                new_len = best_prev_len + 1
                D[i][j] = new_len
                prev[i][j] = best_k
                
                if new_len > best_len:
                    best_len = new_len
                    best_i = i
                    best_j = j
            else:
                # No valid predecessor k. Base case: sequence is just P[i], P[j]
                # Length 2.
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

# EVOLVE-BLOCK-START
# The code above is the implementation.
# I will wrap it in the required format.
# EVOLVE-BLOCK-END
