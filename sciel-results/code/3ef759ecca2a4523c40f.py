import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 50
    
    N_MIN = 10000
    N_MAX = 1000000
    
    # Try to find consecutive P-smooth numbers for small P
    # This could yield a much better score if such pairs exist
    # Check P=5, 7, 11
    
    def find_smooth_pair(P):
        # Generate all P-smooth numbers <= N_MAX + 1
        primes = []
        for i in range(2, P+1):
            # Check if i is prime
            is_p = True
            for j in range(2, int(i**0.5)+1):
                if i % j == 0:
                    is_p = False
                    break
            if is_p:
                primes.append(i)
                
        if not primes:
            return None
            
        # Generate smooth numbers
        # Use a set for O(1) lookup
        smooth_nums = set()
        
        # Recursive generation
        def gen(idx, current_val):
            if current_val > N_MAX + 1:
                return
            smooth_nums.add(current_val)
            if idx < len(primes):
                p = primes[idx]
                next_val = current_val
                while next_val <= N_MAX + 1:
                    gen(idx + 1, next_val)
                    next_val *= p
                    if next_val > N_MAX + 1:
                        break
                # Also allow skipping this prime (handled by recursion structure? No, standard generation includes all combinations)
                # The above recursion generates numbers of form prod(p_i^e_i) where e_i >= 0.
                # Wait, the standard way is to multiply by p and recurse on same prime?
                # No, the above `gen(idx+1, next_val)` moves to next prime.
                # To get all powers, we need to loop on the same prime.
                
        # Let's use a simpler iterative approach with a list of candidates
        # Start with 1
        candidates = [1]
        for p in primes:
            new_candidates = []
            for c in candidates:
                val = c
                while val <= N_MAX + 1:
                    new_candidates.append(val)
                    val *= p
            candidates = new_candidates
            
        # candidates now contains all P-smooth numbers <= N_MAX+1
        # Sort them
        candidates.sort()
        
        # Check for consecutive
        for i in range(len(candidates) - 1):
            if candidates[i+1] - candidates[i] == 1:
                n = candidates[i]
                if n >= N_MIN:
                    return n
        return None

    # Try P=5
    n_found = find_smooth_pair(5)
    if n_found is not None:
        return n_found
        
    # Try P=7
    n_found = find_smooth_pair(7)
    if n_found is not None:
        return n_found
        
    # Try P=11
    n_found = find_smooth_pair(11)
    if n_found is not None:
        return n_found
        
    # If no small P pair found, fall back to full scan for min LPF
    # This will find the global minimum, likely 13
    
    limit = N_MAX + 1
    is_prime = np.ones(limit + 1, dtype=np.bool_)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i : limit+1 : i] = False
    
    primes = np.where(is_prime)[0]
    
    lpf = np.zeros(limit + 1, dtype=np.int64)
    for p in primes:
        multiples = np.arange(p, limit + 1, p)
        lpf[multiples] = p
    
    if time.time() > deadline - 5:
        return 10000
        
    n_range = np.arange(N_MIN, N_MAX + 1)
    lpf_n = lpf[n_range]
    lpf_np1 = lpf[n_range + 1]
    scores = np.maximum(lpf_n, lpf_np1)
    
    best_idx = np.argmin(scores)
    best_n = int(n_range[best_idx])
    
    return best_n
