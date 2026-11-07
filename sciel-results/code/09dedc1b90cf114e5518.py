"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin

    # Primes to use for generating smooth numbers
    # Using primes up to 47
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    
    max_k = 10**9
    max_m = 10**18
    
    # Dictionary to store lists of k for each M
    m_to_ks = defaultdict(list)
    
    # Recursive generation of smooth numbers
    # idx: index of current prime
    # k: current value of k
    # sigma_k: current value of sigma(k)
    def gen_smooth(idx, k, sigma_k):
        if time.time() > deadline:
            return
            
        if idx == len(primes):
            # We have considered all primes, k is a valid smooth number
            M = k * sigma_k
            if M <= max_m:
                m_to_ks[M].append(k)
            return

        p = primes[idx]
        p_pow = 1
        sigma_p = 1
        for e in range(0, 30):
            new_k = k * p_pow
            if new_k > max_k:
                break
            
            new_sigma = sigma_k * sigma_p
            
            # Prune if M exceeds limit
            if new_k * new_sigma > max_m:
                break
            
            # Recurse to next prime
            gen_smooth(idx + 1, new_k, new_sigma)
            
            # Update p_pow and sigma_p for next exponent
            p_pow *= p
            sigma_p += p_pow

    gen_smooth(0, 1, 1)
    
    # Find the best M
    best_m = 0
    best_ks = []
    for M, ks in m_to_ks.items():
        if len(ks) > len(best_ks):
            best_m = M
            best_ks = ks
            
    if not best_ks:
        # Fallback to known solution
        return {'M': 333312, 'k': [336, 372, 434]}
        
    # Ensure types are Python int
    best_ks = [int(k) for k in best_ks]
    
    return {'M': int(best_m), 'k': best_ks}
# EVOLVE-BLOCK-END
