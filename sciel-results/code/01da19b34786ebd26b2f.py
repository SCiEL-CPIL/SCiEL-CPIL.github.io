import math
import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Primes to consider
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    max_exp = 4
    limit_M = 10**18
    
    # Precompute sigma(p^e)
    # sigma(p^e) = 1 + p + ... + p^e = (p^(e+1)-1)/(p-1)
    # We will compute on the fly or precompute for small e
    
    M_counts = defaultdict(list)
    
    # Recursive search
    # state: index of prime, current k, current sigma
    def search(idx, current_k, current_sigma):
        if time.time() > deadline:
            return
            
        if idx == len(primes):
            if current_k > 1: # k=1 gives M=1, trivial
                M_val = current_k * current_sigma
                if M_val <= limit_M:
                    M_counts[M_val].append(current_k)
            return
            
        p = primes[idx]
        
        # Option 1: exponent 0
        search(idx + 1, current_k, current_sigma)
        
        # Option 2: exponent >= 1
        # k_factor = p^e
        # sigma_factor = 1 + p + ... + p^e
        k_factor = 1
        sigma_factor = 1
        for e in range(1, max_exp + 1):
            k_factor *= p
            sigma_factor += k_factor # sigma_factor becomes 1 + p + ... + p^e
            
            new_k = current_k * k_factor
            new_sigma = current_sigma * sigma_factor
            
            new_M = new_k * new_sigma
            if new_M > limit_M:
                break
                
            search(idx + 1, new_k, new_sigma)
            
    search(0, 1, 1)
    
    # Find best M
    best_M = 1
    best_ks = [1]
    max_count = 1
    
    for M, ks in M_counts.items():
        if len(ks) > max_count:
            max_count = len(ks)
            best_M = M
            best_ks = ks
            # Optional: early exit if we find a very high count?
            # But we want to maximize, so we should check all.
            # However, if time is running out, we should return current best.
            if time.time() > deadline - 5.0:
                break
                
    # Ensure types are correct
    # M is int, k is list of ints
    # Check distinctness
    best_ks = list(set(best_ks))
    
    # Verify one more time to be safe
    # The task says "checked with exact factorization" by evaluator.
    # Our generation ensures k * sigma(k) = M.
    
    return {'M': int(best_M), 'k': [int(k) for k in best_ks]}
