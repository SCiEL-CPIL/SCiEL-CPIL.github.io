import time
import math
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Primes to use for smooth numbers
    # Using primes up to 47 (15 primes)
    # 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    num_primes = len(primes)
    
    M_to_ks = defaultdict(list)
    best_M = 1
    max_count = 1
    
    # Recursive generation of smooth numbers
    # idx: index of the next prime to consider
    # k: current product
    # sigma: current sigma(k)
    # limit: 10^18
    limit = 10**18
    
    def generate(idx, k, sigma):
        nonlocal best_M, max_count
        
        # Check time
        if time.time() > deadline:
            return
            
        # Calculate M = k * sigma(k)
        # If M <= limit, record it
        M = k * sigma
        if M <= limit:
            M_to_ks[M].append(k)
            if len(M_to_ks[M]) > max_count:
                max_count = len(M_to_ks[M])
                best_M = M
        
        # If no more primes to add, stop
        if idx >= num_primes:
            return
            
        p = primes[idx]
        
        # Try adding p^a for a = 1, 2, ...
        # k_new = k * p^a
        # sigma_new = sigma * (1 + p + ... + p^a)
        
        p_pow = p
        sigma_p_pow = 1 + p
        
        # We can multiply k by p^a as long as k * p^a <= limit
        # And M = k * p^a * sigma * sigma_p_pow <= limit
        # So k * p^a * sigma * sigma_p_pow <= 10^18
        
        # Since sigma >= 1 and sigma_p_pow >= 2, 
        # k * p^a must be <= limit / 2
        
        # Let's just iterate while k * p_pow <= limit // 2 (safe margin for sigma)
        # Actually, we can just check M <= limit inside the loop
        
        while True:
            k_new = k * p_pow
            if k_new > limit: 
                break
            
            sigma_new = sigma * sigma_p_pow
            M_new = k_new * sigma_new
            
            if M_new > limit:
                # Since p_pow increases, M_new will only increase
                # So we can break
                break
                
            # Recurse with next prime
            generate(idx + 1, k_new, sigma_new)
            
            # Update for next power
            # p_pow *= p
            # sigma_p_pow = sigma_p_pow * p + 1
            
            # Check overflow for p_pow
            if p_pow > limit // p:
                break
            p_pow *= p
            sigma_p_pow = sigma_p_pow * p + 1
            
            # If sigma_p_pow is too large, M_new will exceed limit
            # But the loop condition handles it.
            
            # Time check inside loop to be safe
            if time.time() > deadline:
                return

    # Start with k=1, sigma(1)=1
    generate(0, 1, 1)
    
    # If no better solution found, return 1
    if max_count == 1:
        return {'M': 1, 'k': [1]}
        
    # Retrieve best k's
    # Ensure distinct (they should be, as each k is generated uniquely by its prime factorization)
    # But set conversion is safe and cheap for <1000 items
    k_list = list(set(M_to_ks[best_M]))
    k_list.sort()
    
    # Cap at 1000 if somehow more (though we want max, but limit is 1000)
    # The problem says "list of at most 1000 distinct ints"
    # If we have more, we should return 1000.
    # But the score is the number of listed solutions.
    # So we should return min(1000, len(k_list)).
    # However, the task says "The score is the number of listed solutions".
    # And "k is a list of at most 1000 distinct ints".
    # So if we found 1005 solutions, we should list 1000.
    # But wait, "Maximize" the score.
    # If f(M) > 1000, the score is 1000.
    # So we just need to return 1000 of them.
    
    if len(k_list) > 1000:
        k_list = k_list[:1000]
        
    return {'M': int(best_M), 'k': k_list}
