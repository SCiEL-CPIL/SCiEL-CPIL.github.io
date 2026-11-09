# EVOLVE-BLOCK-START
import math
import time
from collections import defaultdict

def construct():
    deadline = time.time() + 130  # Leave 20s safety margin
    
    # Sieve-based approach: compute f(k) = k * sigma(k) for k up to N
    # Group by M value, find M with most k's where k <= sqrt(M)
    
    # Start with a moderate N, expand if time permits
    # For k up to 10^6, we can compute sigma via sieve quickly
    
    N = 2_000_000  # Start with 2 million
    
    # Sieve for smallest prime factor to compute sigma efficiently
    spf = list(range(N + 1))  # smallest prime factor
    for i in range(2, int(math.isqrt(N)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, N + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Compute sigma(n) for all n using the spf sieve
    # sigma is multiplicative. We can compute it via:
    # For each prime p, multiply sigma by (1 + p + p^2 + ... + p^a) for p^a || n
    
    # Better: use a sieve-like approach
    # sigma[n] = sum of divisors of n
    # We can compute this using: for each d, add d to sigma[multiple of d]
    # This is O(N log N)
    
    sigma = [0] * (N + 1)
    for d in range(1, N + 1):
        for m in range(d, N + 1, d):
            sigma[m] += d
    
    # Now compute f(k) = k * sigma(k) and group by M
    # But M can be up to ~N * sigma(N) which could be large
    # We need M <= 10^18
    
    M_to_ks = defaultdict(list)
    
    for k in range(1, N + 1):
        M_val = k * sigma[k]
        if M_val > 10**18:
            continue
        # Check k <= sqrt(M_val)
        if k <= math.isqrt(M_val):
            M_to_ks[M_val].append(k)
    
    # Find M with most solutions
    best_M = 333312
    best_ks = [336, 372, 434]
    best_count = 3
    
    for M_val, ks in M_to_ks.items():
        count = len(ks)
        if count > best_count:
            best_count = count
            best_M = M_val
            best_ks = ks
    
    # If we have time, try larger N
    remaining = deadline - time.time()
    if remaining > 30 and best_count < 10:
        # Try to extend the sieve
        N2 = 5_000_000
        # Recompute with larger N
        sigma2 = [0] * (N2 + 1)
        for d in range(1, N2 + 1):
            for m in range(d, N2 + 1, d):
                sigma2[m] += d
        
        M_to_ks2 = defaultdict(list)
        for k in range(1, N2 + 1):
            M_val = k * sigma2[k]
            if M_val > 10**18:
                continue
            if k <= math.isqrt(M_val):
                M_to_ks2[M_val].append(k)
        
        for M_val, ks in M_to_ks2.items():
            count = len(ks)
            if count > best_count:
                best_count = count
                best_M = M_val
                best_ks = ks
    
    # Validate and prepare result
    M = int(best_M)
    sqrt_M = math.isqrt(M)
    
    valid_ks = []
    for k in best_ks:
        if 1 <= k <= sqrt_M:
            # Verify k * sigma(k) == M
            if k * sigma[k] == M:
                valid_ks.append(int(k))
    
    valid_ks = sorted(set(valid_ks))
    
    # Fallback
    if not valid_ks:
        M = 333312
        valid_ks = [336, 372, 434]
    
    valid_ks = valid_ks[:1000]
    
    return {'M': M, 'k': valid_ks}
# EVOLVE-BLOCK-END
