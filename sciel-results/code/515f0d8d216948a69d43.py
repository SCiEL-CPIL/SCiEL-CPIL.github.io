import math
import time
from collections import defaultdict

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = 120.0  # Safety margin from 150s
    
    # We will search k up to N
    # Start with N = 10^7
    N = 10**7
    
    # Sieve for Smallest Prime Factor (SPF)
    # Using a list of ints. 10^7 ints is ~80MB in Python?
    # Actually, list of ints in Python is heavy. 
    # Let's use array('I') or similar? Or just list if memory allows.
    # 10^7 * 28 bytes = 280MB. Might be too high.
    # Let's try to be smarter or use smaller N if memory is an issue.
    # But we have 150s.
    
    # Alternative: Use a simpler sieve for sigma directly?
    # sigma = [0] * (N+1)
    # for i in range(1, N+1):
    #     for j in range(i, N+1, i):
    #         sigma[j] += i
    # This is O(N log N). For N=10^7, ~1.6e8 ops.
    # In Python, 1.6e8 iterations is too slow (~10-20s per loop? No, inner loop is fast but total ops high).
    # 1.6e8 simple additions in Python might take 5-10 seconds.
    # Let's try this direct method as it's simpler and avoids SPF complexity.
    # We can optimize by only iterating i up to N, but the inner loop is the cost.
    
    # Let's use a hybrid:
    # 1. Sieve primes.
    # 2. Use formula for sigma based on prime factorization.
    
    # Actually, the direct sigma sieve is easier to implement correctly.
    # Let's estimate time for N=5*10^6 first.
    
    N = 5 * 10**6
    sigma = [0] * (N + 1)
    
    # Direct sieve for sigma
    # Optimization: Use local variables
    sig = sigma
    n_limit = N
    for i in range(1, n_limit + 1):
        # Add i to all multiples of i
        # j = i, 2i, 3i, ...
        for j in range(i, n_limit + 1, i):
            sig[j] += i
            
    # Check time
    if time.time() - start_time > deadline - 10:
        # Fallback to seed if too slow (should not happen for 5e6)
        pass

    # Now group k by M = k * sigma(k)
    groups = defaultdict(list)
    max_M = 10**18
    
    for k in range(1, N + 1):
        m = k * sig[k]
        if m <= max_M:
            groups[m].append(k)
        else:
            # Since k is increasing and sigma(k) >= k, m is generally increasing?
            # Not strictly, sigma(k) fluctuates.
            # But for large k, m is large.
            # We can break if k * k > max_M? 
            # Since sigma(k) >= k, m >= k^2.
            # So if k^2 > max_M, we can break.
            if k * k > max_M:
                break

    # Find best group
    best_M = 0
    best_k_list = []
    
    for m, k_list in groups.items():
        # We already ensured m <= 10^18
        # Condition k <= sqrt(m) is always true as proven.
        if len(k_list) > len(best_k_list):
            best_M = m
            best_k_list = k_list
            
    # If we have time, we can try larger N?
    # For now, return best found.
    # To ensure we beat score 3, N=5e6 should be enough to find collisions.
    # Collisions are frequent for smooth numbers.
    
    # Convert to required types
    # k_list is a list of ints.
    # best_M is an int.
    
    return {'M': int(best_M), 'k': [int(x) for x in best_k_list]}
