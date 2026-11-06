import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 50
    
    N_MIN = 10000
    N_MAX = 1000000
    
    # Try to find consecutive P-smooth numbers for small P
    # This could yield a much better score if such pairs exist
    # Check P=5, 7, 11, 13
    
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
            
        # Generate all P-smooth numbers <= N_MAX + 1 using BFS/iterative approach
        smooth_nums = set([1])
        changed = True
        while changed:
            changed = False
            current_list = list(smooth_nums)
            for num in current_list:
                for p in primes:
                    new_num = num * p
                    if new_num <= N_MAX + 1 and new_num not in smooth_nums:
                        smooth_nums.add(new_num)
                        changed = True
        
        # Convert to sorted list
        candidates = sorted(smooth_nums)
        
        # Check for consecutive pairs
        for i in range(len(candidates) - 1):
            if candidates[i+1] - candidates[i] == 1:
                n = candidates[i]
                if n >= N_MIN and n <= N_MAX:
                    return n
        return None

    # Try small P values - if we find consecutive P-smooth numbers, 
    # the largest prime factor is at most P
    for P in [5, 7, 11, 13]:
        n_found = find_smooth_pair(P)
        if n_found is not None:
            return n_found
        
    # If no small P pair found, fall back to full scan for min LPF
    # This will find the global minimum
    
    limit = N_MAX + 1
    
    # Sieve for primes
    is_prime = np.ones(limit + 1, dtype=np.bool_)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i : limit+1 : i] = False
    
    primes = np.where(is_prime)[0]
    
    # Compute LPF (largest prime factor) for each number
    lpf = np.zeros(limit + 1, dtype=np.int64)
    for p in primes:
        multiples = np.arange(p, limit + 1, p)
        lpf[multiples] = p
    
    if time.time() > deadline - 5:
        return 10000
        
    # Scan all candidates
    n_range = np.arange(N_MIN, N_MAX + 1)
    lpf_n = lpf[n_range]
    lpf_np1 = lpf[n_range + 1]
    scores = np.maximum(lpf_n, lpf_np1)
    
    best_idx = np.argmin(scores)
    best_n = int(n_range[best_idx])
    
    return best_n
