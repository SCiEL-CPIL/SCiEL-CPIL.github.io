import math
from collections import defaultdict

def construct():
    LIMIT_M = 10**18
    
    # Primes to use. 
    # Using more primes increases the combinatorial space and chance of collisions.
    # However, each prime adds a dimension to the search.
    # Let's try primes up to 23.
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    # Precompute (T, k_part) for each prime and valid exponent a
    # T = p^a * sigma(p^a)
    # k_part = p^a
    prime_data = []
    for p in primes:
        options = []
        a = 0
        p_pow = 1
        sigma_val = 1
        while True:
            T = p_pow * sigma_val
            if T > LIMIT_M:
                break
            options.append((T, p_pow))
            # Increment a
            a += 1
            p_pow *= p
            if p_pow > LIMIT_M: # p^a cannot exceed M
                break
            sigma_val += p_pow
            # Check if T exceeds limit for next iteration? 
            # We check at the start of the loop for the current a.
            # But we need to ensure the *next* T is computed correctly.
            # The loop structure above computes T for current a, then increments.
            # The check `if T > LIMIT_M` at the start handles the break.
            # However, `p_pow` and `sigma_val` are updated for the *next* a.
            # We need to make sure we don't overflow or go too far.
            # Since LIMIT_M is 1e18, and p >= 2, a is small (< 60).
            
        prime_data.append(options)
        
    # Sort primes by the minimum T (which is 1 for a=0, so this doesn't help much)
    # Better to sort by the "growth factor" or just keep them in increasing order of p.
    # Smaller p -> smaller T for a>0 -> allows more combinations.
    # So [2, 3, 5...] is good.
    
    m_to_ks = defaultdict(set)
    
    def dfs(idx, curr_M, curr_k):
        if curr_M > LIMIT_M:
            return
            
        if idx == len(primes):
            if curr_M > 1:
                m_to_ks[curr_M].add(curr_k)
            return
            
        options = prime_data[idx]
        for T, k_part in options:
            new_M = curr_M * T
            if new_M > LIMIT_M:
                # Since options are sorted by a (and thus T is increasing), 
                # we can break if T is increasing.
                # Is T increasing with a?
                # T(a) = p^a (1 + p + ... + p^a). Yes, it is strictly increasing for p>=2, a>=0.
                break
            new_k = curr_k * k_part
            dfs(idx + 1, new_M, new_k)
            
    dfs(0, 1, 1)
    
    best_M = 1
    best_ks = [1]
    
    for M, ks in m_to_ks.items():
        if len(ks) > len(best_ks):
            best_ks = list(ks)
            best_M = M
            
    # Verify constraints
    # k_i <= sqrt(M)
    # As discussed, k * sigma(k) = M => k <= M/k = sigma(k). 
    # sigma(k) >= k + 1 for k > 1. So k < M/k => k^2 < M => k < sqrt(M).
    # For k=1, M=1, 1 <= 1.
    # So all generated k are valid.
    
    # Ensure types are Python int
    best_M = int(best_M)
    best_ks = [int(k) for k in best_ks]
    
    return {'M': best_M, 'k': best_ks}
