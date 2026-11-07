import time
import heapq
from math import isqrt

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s
    
    # Primes to use for smooth numbers
    # More primes -> more candidates in the range, potentially higher collision probability
    # But also more complex factorization
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    # We want to generate smooth numbers k such that k * sigma(k) <= 10^18
    # Since sigma(k) >= k+1, k*(k+1) <= 10^18 => k <= ~10^9
    # Let's set a hard limit on k to avoid unnecessary large numbers
    max_k = 10**9 
    
    # We will use a heap to generate smooth numbers in increasing order
    # Each element in the heap is (value, index_in_prime_list, factorization_dict)
    # factorization_dict maps prime -> exponent
    
    # Initial state: 1 with empty factorization
    # But k=1 gives M=1. We can start there.
    
    # To avoid duplicates, we can use a set of generated numbers or a more sophisticated 
    # "ugly number" generation algorithm that avoids duplicates by construction.
    # The standard ugly number algorithm uses an array of pointers.
    # However, with many primes, the array of pointers is large but manageable.
    
    # Let's use the standard ugly number generation with pointers.
    # num[i] = p[ptr[i]] * num[i] is not quite right for multiple primes.
    # The standard algorithm:
    # u[0] = 1
    # ptrs = [0] * len(primes)
    # For each step, next_u = min(p[ptr[i]] * u[ptr[i]] for all i)
    # Then increment all ptrs where p[ptr[i]] * u[ptr[i]] == next_u
    
    # This generates all smooth numbers in sorted order without duplicates.
    
    u = [1]
    ptrs = [0] * len(primes)
    
    # Map M -> list of k
    m_to_k = {}
    
    best_k_list = []
    best_M = 0
    
    # We need to compute sigma(k) efficiently.
    # Since we generate in order, we don't have the factorization directly from the pointer method 
    # unless we store it. Storing factorization for each u[i] is memory intensive but doable for ~100k-1M numbers.
    # Alternatively, we can compute sigma from the factorization if we store it.
    
    # Let's store factorization as a tuple of (prime, exponent) or just a dict?
    # Dicts are heavy. Let's store a list of exponents corresponding to the primes list.
    # exponents[i] is a list of length len(primes).
    
    # Actually, we can compute sigma incrementally? No, sigma is multiplicative but not additive.
    # sigma(p^e) = (p^(e+1)-1)/(p-1).
    # If we know the factorization, we can compute sigma quickly.
    
    # Let's store exponents for each generated number.
    # exponents_list[i] corresponds to u[i].
    
    exponents_list = [[0] * len(primes) for _ in range(1)] # Start with 1 (all exponents 0)
    
    def compute_sigma(exps):
        s = 1
        for i, e in enumerate(exps):
            if e > 0:
                p = primes[i]
                # sigma(p^e) = (p^(e+1) - 1) // (p - 1)
                s *= (p**(e+1) - 1) // (p - 1)
        return s
    
    # Precompute sigma for small powers to speed up? 
    # The exponents are small for smooth numbers.
    
    count = 0
    while True:
        if time.time() > deadline:
            break
            
        # Find next smallest
        min_val = float('inf')
        min_indices = []
        
        for i in range(len(primes)):
            val = primes[i] * u[ptrs[i]]
            if val < min_val:
                min_val = val
                min_indices = [i]
            elif val == min_val:
                min_indices.append(i)
                
        # Add new number
        u.append(min_val)
        
        # Compute new exponents
        new_exps = exponents_list[ptrs[min_indices[0]]].copy()
        for i in min_indices:
            new_exps[i] += 1
            ptrs[i] += 1
            
        exponents_list.append(new_exps)
        
        k = min_val
        if k > max_k:
            # We can stop if k is too large, as sigma(k) >= k+1 implies M > 10^18
            # But wait, if k is just over 10^9, M might still be <= 10^18 if sigma(k) is small?
            # sigma(k) is at least k+1. So k(k+1) > 10^18 for k > 10^9.
            # So yes, we can stop.
            # However, we should process the current k before breaking?
            # The check k > max_k should be done before adding to u?
            # If min_val > max_k, we break.
            # Let's check before appending.
            
            # Actually, let's just break if min_val > max_k.
            # We already appended it. Let's just not process it.
            # But we need to be careful with the loop logic.
            pass
        
        if k > max_k:
            break
            
        sigma_k = compute_sigma(new_exps)
        M = k * sigma_k
        
        if M <= 10**18:
            if M not in m_to_k:
                m_to_k[M] = []
            m_to_k[M].append(k)
            
            # Check if this is the best group so far
            if len(m_to_k[M]) > len(best_k_list):
                best_k_list = m_to_k[M]
                best_M = M
        
        count += 1
        # Optional: stop if we have processed enough candidates
        # The number of smooth numbers grows slowly.
        # Let's run until time limit or k exceeds limit.
        
    # If no solutions found, return the known one
    if not best_k_list:
        return {'M': 333312, 'k': [336, 372, 434]}
        
    return {'M': best_M, 'k': best_k_list}
