"""Rewritten program using Dynamic Programming with optimized lookups."""
import time
import bisect

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin from 90s limit
    
    # 1. Generate primes <= 10000
    limit = 10000
    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    
    primes = [i for i, prime in enumerate(is_prime) if prime]
    n = len(primes)
    
    if n == 0:
        return []
    if n == 1:
        return primes
    
    # 2. DP Initialization
    # dp[i][j] = length of longest valid subsequence ending with primes[i], primes[j]
    # prev[i][j] = index k such that primes[k], primes[i] is the predecessor
    dp = [[0] * n for _ in range(n)]
    prev = [[-1] * n for _ in range(n)]
    
    # structures[i] holds data to query max dp[k][i] for k < i with gap constraint
    # Each structure is a tuple: (primes_list, suffix_max_vals, suffix_max_indices)
    structures = [None] * n
    
    max_len = 2
    best_i = 0
    best_j = 1
    
    # 3. Main DP Loop
    # end is the index of the last prime in the subsequence
    for end in range(1, n):
        # Check time
        if time.time() > deadline:
            break
            
        # Calculate dp[i][end] for all i < end
        for i in range(0, end):
            delta = primes[end] - primes[i]
            
            if i == 0:
                dp[i][end] = 2
                prev[i][end] = -1
            else:
                # We need max dp[k][i] for k < i such that primes[i] - primes[k] <= delta
                # This is equivalent to primes[k] >= primes[i] - delta
                threshold = primes[i] - delta
                
                struct = structures[i]
                # struct is (primes[:i], suf_max, suf_idx)
                p_list, suf_max, suf_idx = struct
                
                # Binary search for the first index k in p_list where p_list[k] >= threshold
                k_idx = bisect.bisect_left(p_list, threshold)
                
                if k_idx < len(p_list):
                    # The max value is at suf_max[k_idx]
                    val = suf_max[k_idx]
                    idx = suf_idx[k_idx]
                    dp[i][end] = val + 1
                    prev[i][end] = idx
                else:
                    dp[i][end] = 2
                    prev[i][end] = -1
            
            if dp[i][end] > max_len:
                max_len = dp[i][end]
                best_i = i
                best_j = end
        
        # Build structure for index 'end' to be used in future iterations
        # The structure needs to support queries: max dp[k][end] for k in [k_min, end-1]
        # where k_min is determined by a threshold on primes[k].
        
        p_list_end = primes[:end]
        vals = [dp[i][end] for i in range(end)]
        
        suf_max = [0] * end
        suf_idx = [0] * end
        
        if end > 0:
            suf_max[end-1] = vals[end-1]
            suf_idx[end-1] = end-1
            
            for k in range(end-2, -1, -1):
                if vals[k] >= suf_max[k+1]:
                    suf_max[k] = vals[k]
                    suf_idx[k] = k
                else:
                    suf_max[k] = suf_max[k+1]
                    suf_idx[k] = suf_idx[k+1]
        
        structures[end] = (p_list_end, suf_max, suf_idx)
        
        # Early termination if time is running out, though unlikely
        if time.time() > deadline:
            break

    # 4. Reconstruct the sequence
    if max_len < 2:
        return primes[:max_len] # Should not happen if n>=2
    
    seq = [primes[best_j], primes[best_i]]
    i = best_i
    j = best_j
    
    while prev[i][j] != -1:
        k = prev[i][j]
        seq.append(primes[k])
        j = i
        i = k
        
    seq.reverse()
    return seq

if __name__ == "__main__":
    # Simple test
    res = construct()
    print(f"Length: {len(res)}")
    print(f"Sequence: {res}")
    # Verify gaps
    gaps = [res[i+1] - res[i] for i in range(len(res)-1)]
    is_non_decreasing = all(gaps[i] <= gaps[i+1] for i in range(len(gaps)-1))
    print(f"Gaps non-decreasing: {is_non_decreasing}")
