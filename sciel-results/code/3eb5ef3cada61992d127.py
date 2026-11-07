import math
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 110  # 10s margin from 120s limit
    
    # We will try increasing N until time runs out or we find a solution with count > 3
    # Start with a reasonable N, then increase.
    # The sieve O(N log N) is fast in NumPy.
    
    # Pre-define a maximum N we can afford.
    # 10^7 is a good target.
    
    best_M = 333312
    best_k_list = [336, 372, 434]
    best_count = 3
    
    # We can try a few sizes
    target_ns = [2_000_000, 5_000_000, 10_000_000, 20_000_000]
    
    for N in target_ns:
        if time.time() > deadline:
            break
            
        try:
            # 1. Compute sigma(k) for k=1..N using divisor sum sieve
            # sigma = sum of divisors
            sigma = np.zeros(N + 1, dtype=np.int64)
            
            # This loop is O(N log N)
            # For d in 1..N: sigma[d::d] += d
            # This is vectorized.
            for d in range(1, N + 1):
                sigma[d::d] += d
                
            # 2. Compute M = k * sigma(k)
            k_vals = np.arange(1, N + 1, dtype=np.int64)
            M_vals = k_vals * sigma
            
            # 3. Filter M <= 10^18
            mask = M_vals <= 10**18
            M_valid = M_vals[mask]
            k_valid = k_vals[mask]
            
            if len(M_valid) == 0:
                continue
                
            # 4. Find most frequent M
            # np.unique is efficient for sorted arrays, but M_vals is not sorted.
            # We can use a dictionary if N is small, but for N=10^7, dict might be slow/memory heavy.
            # Let's try np.unique.
            unique_M, counts, inverse = np.unique(M_valid, return_counts=True, return_inverse=True)
            
            if len(unique_M) == 0:
                continue
                
            max_idx = np.argmax(counts)
            current_count = counts[max_idx]
            current_M = int(unique_M[max_idx])
            
            if current_count > best_count:
                best_count = current_count
                best_M = current_M
                # Retrieve k's
                # k_valid is 1-indexed effectively (k_vals starts at 1)
                # inverse maps M_valid indices to unique_M indices
                # We need k_valid where inverse == max_idx
                k_indices = np.where(inverse == max_idx)[0]
                best_k_list = [int(x) for x in k_valid[k_indices]]
                
                # If we found a very high count, we can stop early?
                # But maybe a larger N finds even more?
                # Let's continue to the next N unless we hit the deadline.
                
        except MemoryError:
            # If N is too large, break
            break
        except Exception:
            # Catch other errors
            break
            
    # Ensure types are correct
    # k must be list of ints
    # M must be int
    
    # Check if best_k_list is valid (distinct, <= sqrt(M), etc.)
    # The sieve guarantees k <= N.
    # We must ensure k <= sqrt(M).
    # Since M = k * sigma(k), and sigma(k) >= k+1 (for k>1), M > k^2.
    # So k < sqrt(M) is always true for k>1.
    # For k=1, M=2, sqrt(2)=1.41, k=1 <= 1.41.
    
    # Distinctness is guaranteed by the nature of the search (each k appears once in the array).
    
    return {'M': int(best_M), 'k': best_k_list}
