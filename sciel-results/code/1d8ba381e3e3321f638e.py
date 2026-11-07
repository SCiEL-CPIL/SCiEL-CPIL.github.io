import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    # Use a sieve-based approach to compute sigma(n) for all n up to N
    # sigma(n) is multiplicative: if n = prod p_i^a_i, sigma(n) = prod (p_i^(a_i+1)-1)/(p_i-1)
    # We can compute sigma using a modified sieve
    
    # Start with a reasonable N, increase if time allows
    N = 5_000_000
    
    # sigma_arr[n] will hold sigma(n)
    # Use array of zeros, initialize sigma[1]=1
    sigma_arr = [0] * (N + 1)
    sigma_arr[1] = 1
    
    # For each prime p, for each power p^a, add contribution to multiples
    # More efficient: use the fact that sigma is multiplicative
    # Initialize sigma_arr[n] = 1 for all n, then for each prime p,
    # for each multiple of p, multiply by the appropriate factor
    
    # Actually, let's use a different approach:
    # sigma(n) = sum of divisors. We can compute it as:
    # For each d from 1 to N, for each multiple m of d, add d to sigma[m]
    # This is O(N log N) which is fine for N=5M
    
    # But that's O(N log N) additions. For N=5M, that's about 5M * 16 = 80M operations.
    # In Python, this might be slow. Let's use numpy.
    
    import numpy as np
    
    sigma_arr = np.zeros(N + 1, dtype=np.int64)
    sigma_arr[1] = 1
    
    # For d from 1 to N, add d to sigma[d], sigma[2d], sigma[3d], ...
    # This is the classic divisor sum sieve
    for d in range(1, N + 1):
        if time.time() > deadline - 20:  # Check time, leave 20s for rest
            break
        # Add d to all multiples of d
        multiples = np.arange(d, N + 1, d)
        sigma_arr[multiples] += d
    
    # Now sigma_arr[k] = sigma(k) for k = 1..N
    # Compute M = k * sigma(k) and group
    groups = {}
    limit = 10**18
    
    # Convert to list for faster iteration
    sigma_list = sigma_arr.tolist()
    
    for k in range(1, N + 1):
        if k % 1000000 == 0:
            if time.time() > deadline - 10:
                break
        
        sigma_k = sigma_list[k]
        M = k * sigma_k
        if M <= limit:
            if M in groups:
                groups[M].append(k)
            else:
                groups[M] = [k]
    
    # Find best M
    best_M = 0
    best_k_list = []
    max_count = 0
    
    for M, k_list in groups.items():
        if len(k_list) > max_count:
            max_count = len(k_list)
            best_M = M
            best_k_list = k_list
    
    # Ensure types are correct
    best_M = int(best_M)
    best_k_list = [int(k) for k in best_k_list]
    
    return {'M': best_M, 'k': best_k_list}
