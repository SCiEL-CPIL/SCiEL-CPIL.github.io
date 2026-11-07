import math
import time
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s buffer for safety
    
    # Fallback known solution (from current program's fallback, score 1.0 equivalent? No, current score is 1.6667)
    # The current program's fallback is 3 solutions.
    # We hope to find more.
    fallback_M = 333312
    fallback_ks = [336, 372, 434]
    
    # Try to search with NumPy
    # Start with a moderate N and increase if time permits?
    # Or just pick a large N that fits in time.
    # Let's try N = 20,000,000 first to be safe on memory/time, then maybe 50M.
    # 20M * 8 bytes = 160MB. Safe.
    # 50M * 8 bytes = 400MB. Safe.
    
    N = 50_000_000
    
    # Check if we have enough time
    if time.time() > deadline - 60.0:
        return {'M': fallback_M, 'k': fallback_ks}
        
    try:
        # Initialize sigma array
        # sigma[0] is unused, sigma[1] = 1
        sigma = np.ones(N + 1, dtype=np.int64)
        
        # Sieve for sigma
        # for i in range(1, N + 1):
        #    sigma[i::i] += i
        # This loop is still O(N log N) in Python loop iterations, but inner is C.
        # However, the Python loop itself runs N times.
        # N=50M iterations in Python loop might take ~5-10s just for the loop overhead.
        # Plus the vectorized ops.
        # Let's optimize the sieve loop.
        
        # We can skip even numbers? No, sigma is needed for all.
        # We can precompute primes? No, sigma needs all divisors.
        
        # Alternative:
        # Use a faster sieve variant?
        # The standard "add i to multiples" is the way.
        
        # Let's try to reduce N if we are worried.
        # Actually, the Python loop `for i in range(1, N+1)` is the bottleneck if N is large.
        # 50M iterations in Python is ~5-10s.
        # Each iteration does a vectorized slice add.
        # For small i, the slice is large. For large i, the slice is small.
        # Total work is N log N.
        
        # Let's try N = 20_000_000 to be safer on time.
        N = 20_000_000
        sigma = np.ones(N + 1, dtype=np.int64)
        
        # To speed up the Python loop, we can't really avoid it.
        # But we can check time periodically.
        
        for i in range(1, N + 1):
            if time.time() > deadline - 20.0:
                break
            sigma[i::i] += i
            
        if time.time() > deadline - 10.0:
            return {'M': fallback_M, 'k': fallback_ks}
            
        # Compute M = k * sigma[k]
        # k = 1..N
        k_arr = np.arange(1, N + 1, dtype=np.int64)
        M_arr = k_arr * sigma[1:]
        
        # Filter M <= 10^18
        # Since sigma[k] >= k+1, M >= k(k+1).
        # k ~ 10^9 gives M ~ 10^18.
        # Our N is 2*10^7, so k is small. M will be well below 10^18.
        # So no filtering needed for upper bound, but good to be safe.
        # Actually, we want to find collisions.
        
        # Find the M with the most occurrences.
        # We can use np.unique with return_counts?
        # But we need the keys k.
        # np.unique returns sorted unique values.
        # We need to map back to k.
        
        # Method:
        # 1. Sort M_arr and keep track of original indices (k).
        # 2. Find the longest run of identical M values.
        
        # argsort M_arr
        # This is O(N log N). 20M elements.
        # Sorting 20M int64s takes ~1-2s in NumPy.
        
        if time.time() > deadline - 10.0:
            return {'M': fallback_M, 'k': fallback_ks}
            
        order = np.argsort(M_arr)
        sorted_M = M_arr[order]
        sorted_k = k_arr[order]
        
        # Find runs
        # Compare sorted_M[i] with sorted_M[i+1]
        # Changes in value mark the end of a run.
        
        # Vectorized run finding:
        # diff = np.diff(sorted_M)
        # breaks = np.where(diff != 0)[0] + 1
        # starts = np.concatenate(([0], breaks))
        # ends = np.concatenate((breaks, [len(sorted_M)]))
        # lengths = ends - starts
        
        # Find max length
        # max_len = np.max(lengths)
        # idx = np.argmax(lengths)
        # best_start = starts[idx]
        # best_end = ends[idx]
        # best_ks = sorted_k[best_start:best_end]
        # best_M = sorted_M[best_start]
        
        diff = np.diff(sorted_M)
        breaks = np.where(diff != 0)[0] + 1
        starts = np.concatenate(([0], breaks))
        ends = np.concatenate((breaks, [len(sorted_M)]))
        lengths = ends - starts
        
        max_len = np.max(lengths)
        
        if max_len < len(fallback_ks):
            return {'M': fallback_M, 'k': fallback_ks}
            
        idx = np.argmax(lengths)
        best_start = starts[idx]
        best_end = ends[idx]
        best_ks = sorted_k[best_start:best_end].tolist()
        best_M = int(sorted_M[best_start])
        
        # Verify
        # The problem requires k <= isqrt(M).
        # As derived before, k * sigma(k) = M implies k < sqrt(M) always.
        # So this is satisfied.
        
        # Verify types
        if not isinstance(best_M, int) or not isinstance(best_ks, list):
             return {'M': fallback_M, 'k': fallback_ks}
             
        # Check if best_M <= 10^18
        if best_M > 10**18:
             return {'M': fallback_M, 'k': fallback_ks}
             
        # Check distinctness (guaranteed by construction from unique k's)
        # Check k_i * sigma(k_i) == M for all k_i?
        # We computed M_arr = k * sigma[k], so it should hold.
        # But let's do a quick sanity check on the first few?
        # No, trust the computation.
        
        return {'M': best_M, 'k': best_ks}
        
    except Exception:
        return {'M': fallback_M, 'k': fallback_ks}
