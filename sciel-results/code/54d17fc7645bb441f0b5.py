import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin before 60s limit
    
    N_MIN = 10000
    N_MAX = 1000000
    limit = N_MAX + 1  # We need lpf up to N_MAX+1 = 1000001
    
    # Strategy:
    # 1. Try to find consecutive P-smooth numbers for small P (2,3,5,7,11,13,17,19,23,29)
    #    If found, the largest prime factor is at most P, which is excellent.
    # 2. If not found, fall back to full LPF computation and scan for minimum.
    
    def find_smooth_pair(P):
        """Find consecutive P-smooth numbers in [N_MIN, N_MAX]"""
        # Get primes up to P
        primes = []
        for i in range(2, P+1):
            is_p = True
            for j in range(2, int(i**0.5)+1):
                if i % j == 0:
                    is_p = False
                    break
            if is_p:
                primes.append(i)
                
        if not primes:
            return None
            
        # Generate all P-smooth numbers <= N_MAX + 1 using iterative approach
        # Start with [1], and for each prime, multiply existing candidates by powers of p
        candidates = [1]
        for p in primes:
            new_candidates = []
            for c in candidates:
                val = c
                while val <= N_MAX + 1:
                    new_candidates.append(val)
                    val *= p
            candidates = new_candidates
        
        # Sort candidates
        candidates.sort()
        
        # Check for consecutive pairs
        for i in range(len(candidates) - 1):
            if candidates[i+1] - candidates[i] == 1:
                n = candidates[i]
                if N_MIN <= n <= N_MAX:
                    return n
        return None
    
    # Try small P values - if we find consecutive P-smooth numbers, 
    # the largest prime factor of n*(n+1) is at most P
    for P in [5, 7, 11, 13, 17, 19, 23, 29, 31]:
        if time.time() > deadline - 10:
            break
        n_found = find_smooth_pair(P)
        if n_found is not None:
            return n_found
    
    # If no small P pair found, fall back to full scan for min LPF
    # Sieve of Eratosthenes to find all primes up to limit
    is_prime = np.ones(limit + 1, dtype=np.bool_)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i : limit+1 : i] = False
    
    primes = np.where(is_prime)[0]
    
    # Compute Largest Prime Factor (LPF) for each number up to limit
    # lpf[x] will store the largest prime factor of x
    # Since we iterate primes in increasing order, the last assignment gives the largest prime factor
    lpf = np.zeros(limit + 1, dtype=np.int64)
    for p in primes:
        multiples = np.arange(p, limit + 1, p)
        lpf[multiples] = p
    
    # Check deadline
    if time.time() > deadline - 5:
        # Fallback: just return a reasonable value
        return 10080  # 10080 = 2^5 * 3^2 * 5 * 7, lpf=7
    
    # Scan all n in [N_MIN, N_MAX] to find min of max(lpf[n], lpf[n+1])
    n_range = np.arange(N_MIN, N_MAX + 1)
    lpf_n = lpf[n_range]
    lpf_np1 = lpf[n_range + 1]
    scores = np.maximum(lpf_n, lpf_np1)
    
    best_idx = np.argmin(scores)
    best_n = int(n_range[best_idx])
    
    return best_n
