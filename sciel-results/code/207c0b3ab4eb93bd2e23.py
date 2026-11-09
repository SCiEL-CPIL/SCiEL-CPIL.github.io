import time
import math
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin before 120s/150s limit
    
    # Strategy:
    # 1. Use NumPy to compute phi for a large range [1, N] using a segmented-like approach
    #    (though for contiguous [1,N] a single array is fine if memory allows).
    #    N=50M int64 is 400MB. Might be tight but let's try 20M first to be safe on memory,
    #    or use int32 if possible? phi(n) < n, so for N=20M, int32 is fine (max val ~20M).
    #    Actually, phi values can be up to N. For N=50M, int32 is fine (max 2^31-1 ~ 2.1B).
    #    Let's try N = 30,000,000 with int32 to save memory and increase speed.
    
    N = 30000000
    dtype = np.int32
    
    # Initialize phi array
    phi = np.arange(N + 1, dtype=dtype)
    
    # Sieve for phi
    # Optimization: Only iterate up to sqrt(N) for the outer loop? 
    # No, the standard sieve requires checking all i up to N where phi[i]==i (primes).
    # But we can break early if i*i > N? No, because primes > sqrt(N) still need to be processed
    # to update phi for their multiples (which are just themselves, since 2*p > N).
    # Wait, if p > sqrt(N), the only multiple of p in [1, N] is p itself.
    # phi[p] starts as p. We do phi[p] -= phi[p]//p => p - 1.
    # So for primes p > sqrt(N), we just need to set phi[p] = p - 1.
    # The standard loop `for j in range(i, N+1, i)` for i > sqrt(N) only touches j=i.
    # So we can split the loop:
    # 1. For i from 2 to sqrt(N): standard sieve step.
    # 2. For i from sqrt(N)+1 to N: if phi[i] == i (prime), set phi[i] = i - 1.
    
    limit = int(math.isqrt(N)) + 1
    
    # Phase 1: Sieve for i up to sqrt(N)
    for i in range(2, limit):
        if phi[i] == i:  # i is prime
            # Vectorized update for multiples
            # phi[j] -= phi[j] // i
            # This is a bit tricky to vectorize directly with integer division in a single line
            # without creating a large temporary array, but NumPy handles it.
            # However, `phi[i::i]` creates a view.
            # `phi[i::i] -= phi[i::i] // i` works.
            phi[i::i] -= phi[i::i] // i
            
    # Phase 2: Handle primes > sqrt(N)
    # For i in range(limit, N+1):
    #   if phi[i] == i: phi[i] = i - 1
    # We can do this with a boolean mask.
    # mask = (phi == np.arange(N+1)) # This is expensive to compute for the whole array?
    # Actually, phi[i] == i identifies primes that haven't been touched by smaller primes.
    # For i > sqrt(N), if phi[i] == i, it's prime.
    # Let's find indices where phi == index.
    # indices = np.where(phi == np.arange(N+1))[0]
    # But np.arange(N+1) creates a new array.
    # We can just iterate? No, that's slow in Python.
    # Let's stick to the simple loop for Phase 1 and see if time is okay.
    # Actually, the vectorized step in Phase 1 is the bottleneck.
    # For N=30M, the number of primes up to sqrt(30M) ~ 5477 is small.
    # The vectorized operations are fast.
    
    # Let's refine Phase 2 using NumPy:
    # We need to update phi[i] for all i where phi[i] == i.
    # This is equivalent to: phi = np.where(phi == np.arange(N+1), phi - 1, phi)
    # But np.arange(N+1) is 30M elements. This is fast in NumPy.
    
    # Check time
    if time.time() - start_time > deadline - 10:
        return {'start': 1, 'length': 1}
    
    # Apply Phase 2 update
    # Note: phi[0] and phi[1] are 0 and 1.
    # phi[0]==0, phi[1]==1.
    # For i=0, phi[0]=0. 0==0 is True. phi[0] becomes -1? No, we only care about i>=1.
    # We can just apply it and fix index 0 if needed, or mask it.
    # Let's create a mask of indices 1..N
    idx = np.arange(N + 1)
    mask = (phi == idx)
    # We don't want to modify index 0? phi(0) is undefined usually, but we start from 1.
    # phi[0] becomes -1. It won't be used in the window scan starting at 1.
    phi[mask] -= 1
    
    # Verify: For a prime p > sqrt(N), phi[p] was p. Now it is p-1. Correct.
    # For a composite n > sqrt(N), it must have a prime factor < sqrt(N), so phi[n] was already reduced.
    # So phi[n] != n. Thus mask is False. Correct.
    
    # Now find the longest run of distinct totients
    # Sliding window approach
    # We need to track counts of phi values in the current window [l, r]
    # phi values are in [1, N].
    # We can use a dictionary or an array.
    # An array of size N+1 is 30M integers. ~240MB (int64) or 120MB (int32).
    # Using a dict might be slower due to hashing but saves memory if window is small?
    # But window can be up to 200. So dict is small.
    # However, checking `if val in dict` is O(1) but with higher constant than array access.
    # Let's try array for speed, if memory allows.
    # Total memory: phi (120MB) + counts (120MB) + others. ~300MB. Should be okay.
    
    counts = np.zeros(N + 1, dtype=np.int32)
    
    best_k = 1
    best_s = 1
    
    l = 1
    # r will go from 1 to N
    # We can't easily vectorize the sliding window because of the dynamic shrinking logic.
    # We must iterate r in Python.
    # This will be slow for 30M iterations in pure Python.
    # 30M iterations * ~100ns = 3 seconds? Maybe more like 5-10 seconds.
    # Let's optimize the inner loop.
    
    # Instead of a while loop inside, we can just check if counts[phi[r]] > 0.
    # If yes, we need to move l forward until the previous occurrence is removed.
    # This is the standard two-pointer technique.
    
    # To speed up the Python loop, we can try to process in chunks or use PyPy-like optimizations?
    # No, we are in CPython.
    # Let's just run it and see.
    
    # Optimization: If best_k reaches 200, we can stop.
    # Also, we can check time periodically.
    
    # Let's use a list for counts if numpy array access is slower?
    # numpy array access in Python loop is slower than list access?
    # Actually, `counts[val]` where counts is a list is faster than numpy array in a Python loop.
    # Let's use a list for counts.
    # But initializing a list of 30M zeros takes time and memory.
    # [0] * (N+1) is fast.
    
    counts_list = [0] * (N + 1)
    
    # Convert phi to list? No, phi is numpy array. Accessing phi[r] in Python loop is okay.
    # But converting phi to list takes time.
    # Let's keep phi as numpy array.
    
    l = 1
    # We iterate r from 1 to N
    # To avoid checking time every iteration, we can check every 1000 iterations.
    
    for r in range(1, N + 1):
        val = int(phi[r])  # Convert to int for list indexing
        
        if counts_list[val] > 0:
            # Shrink window from left
            while l < r:
                left_val = int(phi[l])
                counts_list[left_val] -= 1
                l += 1
                if left_val == val:
                    break
        
        counts_list[val] += 1
        
        current_k = r - l + 1
        if current_k > best_k:
            best_k = current_k
            best_s = l
            
            if best_k >= 200:
                break
        
        # Time check every 10000 iterations
        if r % 10000 == 0 and time.time() - start_time > deadline - 5:
            break
            
    return {'start': best_s, 'length': best_k}
# EVOLVE-BLOCK-END
