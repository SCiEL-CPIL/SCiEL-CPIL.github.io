import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Sieve for primes up to 10000
    limit = 10000
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    
    primes = [i for i, v in enumerate(is_prime) if v]
    N = len(primes)
    
    if N == 0:
        return []
        
    INF = 10**9
    
    # dp[i][l] = min last gap for sequence of length l ending at i
    # We use NumPy arrays for speed
    # Max length is N, but practically small. We allocate N+1 to be safe.
    dp = np.full((N, N + 1), INF, dtype=np.int32)
    parent = np.full((N, N + 1), -1, dtype=np.int32)
    
    for i in range(N):
        dp[i, 1] = 0
        parent[i, 1] = -1
        
    # Track max length achieved so far for each index i
    # This helps limit the searchsorted range
    max_len_at = np.zeros(N, dtype=np.int32)
    max_len_at[:] = 1
    
    best_len = 1
    best_index = 0
    
    primes_arr = np.array(primes, dtype=np.int32)
    
    for i in range(1, N):
        if time.time() > deadline:
            break
            
        pi = primes_arr[i]
        
        for j in range(i):
            gap = pi - primes_arr[j]
            
            # We search in the valid, sorted part of dp[j]
            # dp[j, 0] is INF, dp[j, 1] is 0, ...
            # We want the largest l such that dp[j, l] <= gap
            # The array dp[j, 1:max_len_at[j]+1] is sorted non-decreasing.
            
            # If max_len_at[j] is 1, the slice is [0].
            # searchsorted([0], gap, side='right')
            # If gap >= 0, returns 1. max_l = 1.
            
            # If max_len_at[j] is 0 (impossible), returns 0.
            
            # We use max_len_at[j] to limit the search space and ensure we don't hit INFs
            # that might be smaller than gap if we were careless (though INF is large).
            # Actually, just searching the whole row would work if INF > gap, but slicing is safer/faster.
            
            # Optimization: if gap is very small, max_l will be small.
            # But searchsorted is O(log L).
            
            # Create a view of the relevant part
            # Note: slicing a NumPy array creates a view, which is cheap.
            
            # If max_len_at[j] is 1, we know max_l is 1 (since gap >= 0).
            # We can skip searchsorted for small lengths?
            # For now, let's use searchsorted.
            
            # We need to handle the case where the slice is empty (should not happen)
            # Or just let searchsorted handle it.
            
            # To avoid overhead of creating a view for very small slices, we can check
            if max_len_at[j] == 1:
                max_l = 1
            else:
                # Slice from index 1 to max_len_at[j]
                # This corresponds to lengths 1 to max_len_at[j]
                arr_slice = dp[j, 1:max_len_at[j]+1]
                max_l = np.searchsorted(arr_slice, gap, side='right')
                
            if max_l > 0:
                new_len = max_l + 1
                if gap < dp[i, new_len]:
                    dp[i, new_len] = gap
                    parent[i, new_len] = j
                    if new_len > max_len_at[i]:
                        max_len_at[i] = new_len
                        if new_len > best_len:
                            best_len = new_len
                            best_index = i

    # Reconstruct
    if best_len == 1:
        return [primes[best_index]]
        
    path = []
    curr_idx = best_index
    curr_len = best_len
    
    while curr_len > 0 and time.time() < deadline:
        path.append(primes[curr_idx])
        if curr_len == 1:
            break
        prev_idx = parent[curr_idx, curr_len]
        curr_idx = prev_idx
        curr_len -= 1
        
    path.reverse()
    return path
