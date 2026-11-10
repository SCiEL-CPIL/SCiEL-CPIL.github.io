"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import bisect
import numpy as np

def construct():
    start = time.time()
    deadline = start + 50
    
    n_max = 10000
    # Sieve of Eratosthenes using numpy for speed
    is_prime = np.ones(n_max + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n_max**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    primes = [int(p) for p in np.where(is_prime)[0]]
    n = len(primes)
    
    if n <= 2:
        return primes
    
    # dp[j][k] = max length of valid subsequence ending with primes[j], primes[k] (j < k)
    # prev[j][k] = index i such that primes[i] is the element before primes[j] in the best subseq
    dp = np.zeros((n, n), dtype=np.int32)
    prev = np.full((n, n), -1, dtype=np.int32)
    
    best_len = 2
    best_j, best_k = 0, 1
    
    # For each j from 1 to n-1:
    #   Build suffix max of dp[0:j, j] column
    #   For each k > j, use monotonic pointer to find valid i range
    for j in range(1, n):
        if time.time() > deadline:
            break
        
        # Build suffix max: suff_val[i] = max(dp[i][j], dp[i+1][j], ..., dp[j-1][j])
        # suff_arg[i] = argmax
        suff_val = np.zeros(j + 1, dtype=np.int32)
        suff_arg = np.full(j + 1, -1, dtype=np.int32)
        
        col_j = dp[:j, j]
        
        current_max = 0
        current_arg = -1
        for i in range(j - 1, -1, -1):
            if col_j[i] > current_max:
                current_max = col_j[i]
                current_arg = i
            suff_val[i] = current_max
            suff_arg[i] = current_arg
        
        # Monotonic pointer: lo starts at j, moves left as k increases
        # threshold = 2*primes[j] - primes[k] decreases as k increases
        # We need the smallest index lo such that primes[lo] >= threshold
        # As threshold decreases, lo can only decrease (move left)
        lo = j
        
        for k in range(j + 1, n):
            gap_jk = primes[k] - primes[j]
            threshold = 2 * primes[j] - primes[k]
            
            # Move lo left while primes[lo-1] >= threshold
            while lo > 0 and primes[lo - 1] >= threshold:
                lo -= 1
            
            if lo >= j:
                # No valid i exists
                dp[j, k] = 2
                prev[j, k] = -1
            else:
                # Max dp[i][j] for i in [lo, j-1]
                bv = suff_val[lo]
                ba = suff_arg[lo]
                if bv > 0:
                    dp[j, k] = bv + 1
                    prev[j, k] = ba
                else:
                    dp[j, k] = 2
                    prev[j, k] = -1
            
            if dp[j, k] > best_len:
                best_len = dp[j, k]
                best_j, best_k = j, k
    
    if best_len <= 2:
        return primes[:2]
    
    # Reconstruct the sequence
    seq = [primes[best_k], primes[best_j]]
    j, k = best_j, best_k
    while len(seq) < best_len:
        i = prev[j, k]
        if i < 0:
            break
        seq.append(primes[i])
        k = j
        j = i
    
    seq.reverse()
    
    # Validate the result
    if len(seq) < 2:
        return primes[:2]
    
    # Check that all elements are prime and <= 10000
    for p in seq:
        if p > 10000 or not is_prime[p]:
            return primes[:2]
    
    # Check that the sequence is increasing
    for i in range(len(seq) - 1):
        if seq[i] >= seq[i+1]:
            return primes[:2]
    
    # Check that gaps are nondecreasing
    gaps = [seq[i+1] - seq[i] for i in range(len(seq) - 1)]
    for i in range(len(gaps) - 1):
        if gaps[i] > gaps[i+1]:
            return primes[:2]
    
    return seq
# EVOLVE-BLOCK-END
