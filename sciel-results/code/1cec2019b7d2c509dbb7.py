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
    
    # dp[j][k] = length of longest valid subsequence ending with primes[j], primes[k]
    # where the last gap is primes[k] - primes[j]
    # prev[j][k] stores the index i such that the sequence is ... -> i -> j -> k
    # For each (j, k), we need the best i < j such that:
    #   primes[j] - primes[i] <= primes[k] - primes[j]
    #   i.e., primes[i] >= 2*primes[j] - primes[k]
    #   and dp[i][j] is maximized.
    
    # Use numpy arrays for dp and prev for faster access
    dp = np.zeros((n, n), dtype=np.int32)
    prev = np.full((n, n), -1, dtype=np.int32)
    
    best_len = 2
    best_j, best_k = 0, 1
    
    # Convert primes to numpy array for faster bisect? No, bisect works on lists.
    # But we can precompute the primes list for bisect.
    
    for j in range(1, n):
        if time.time() > deadline:
            break
        
        # Build suffix max of dp[i][j] for i = 0..j-1
        # suff_val[i] = max(dp[i][j], dp[i+1][j], ..., dp[j-1][j])
        # suff_arg[i] = argmax
        # We can do this with a simple loop, but let's see if we can vectorize or speed up.
        # Since j is small relative to n initially, and grows, the total work is O(n^2).
        # The inner loop for k is O(n), and the suffix max is O(j). Total O(n^2).
        
        suff_val = np.zeros(j + 1, dtype=np.int32)
        suff_arg = np.full(j + 1, -1, dtype=np.int32)
        
        # dp[:, j] is the column j. We need suffix max of dp[0:j, j].
        col_j = dp[:j, j]
        
        # Compute suffix max and argmax
        # This is a bit tricky to vectorize directly with argmax.
        # Let's do it in a loop for clarity and correctness, as numpy's argmax is global.
        # Actually, we can use a trick: reverse the array, compute cumulative max, then reverse back.
        # But we also need the index.
        
        # Let's stick to the loop for the suffix max construction, it's O(j) and j is up to 1229.
        # The inner loop over k is O(n), so the total is dominated by O(n^2).
        # The constant factor in the suffix max loop is small.
        
        current_max = 0
        current_arg = -1
        for i in range(j - 1, -1, -1):
            if col_j[i] > current_max:
                current_max = col_j[i]
                current_arg = i
            suff_val[i] = current_max
            suff_arg[i] = current_arg
        
        # Now for each k > j
        for k in range(j + 1, n):
            gap_jk = primes[k] - primes[j]
            threshold = primes[j] - gap_jk
            
            # Find the first index lo such that primes[lo] >= threshold
            # bisect_left returns the insertion point.
            lo = bisect.bisect_left(primes, threshold)
            
            if lo >= j:
                dp[j, k] = 2
                prev[j, k] = -1
            else:
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
    return seq
# EVOLVE-BLOCK-END
