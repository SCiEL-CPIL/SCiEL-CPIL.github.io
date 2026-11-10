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
    
    # For each prime index j, maintain a list of Pareto-optimal states:
    # (last_gap, length, prev_idx) sorted by last_gap ascending, with length non-increasing.
    # A state (g1, l1, p1) dominates (g2, l2, p2) if g1 <= g2 and l1 >= l2.
    # We keep only non-dominated states.
    
    # states[j] = list of (gap, length, prev_idx) tuples, sorted by gap
    states = [[] for _ in range(n)]
    
    best_len = 2
    best_j, best_k = 0, 1
    
    # For j=0, no previous states
    # For j>0, we can form a pair (i, j) for any i < j with length 2
    
    for j in range(1, n):
        if time.time() > deadline:
            break
        
        # Build the Pareto frontier for states ending at j
        # A state ending at j with previous i has:
        #   gap = primes[j] - primes[i]
        #   length = 2 + (best extension from i with last_gap <= gap)
        #   prev = i
        
        # We need to compute for each possible previous i < j:
        #   What's the best length of a subsequence ending at i with last_gap <= primes[j] - primes[i]?
        
        # This is complex. Let's use the DP approach but with sparse storage.
        
        # Actually, let's stick with the dense DP but optimize the inner loop.
        pass
    
    # Optimized DP with two-pointer technique for the bisect
    dp = np.zeros((n, n), dtype=np.int32)
    prev = np.full((n, n), -1, dtype=np.int32)
    
    for j in range(1, n):
        if time.time() > deadline:
            break
        
        # Precompute suffix max of dp[i][j] for i in [0, j-1]
        col_j = dp[:j, j]
        
        suff_val = np.zeros(j, dtype=np.int32)
        suff_arg = np.full(j, -1, dtype=np.int32)
        
        current_max = 0
        current_arg = -1
        for i in range(j - 1, -1, -1):
            if col_j[i] > current_max:
                current_max = col_j[i]
                current_arg = i
            suff_val[i] = current_max
            suff_arg[i] = current_arg
        
        # Process all k > j using two-pointer for lo
        # As k increases, threshold = 2*primes[j] - primes[k] decreases
        # So lo = bisect_left(primes, threshold) is non-increasing
        
        lo = 0  # Start at the beginning
        for k in range(j + 1, n):
            threshold = 2 * primes[j] - primes[k]
            
            # Update lo to be the first index where primes[lo] >= threshold
            # Since threshold decreases as k increases, lo can only decrease or stay
            # Actually, lo should be the first index >= threshold
            # As threshold decreases, lo can move left (decrease)
            
            # Use bisect for correctness, but we can optimize
            # For now, just use bisect
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
