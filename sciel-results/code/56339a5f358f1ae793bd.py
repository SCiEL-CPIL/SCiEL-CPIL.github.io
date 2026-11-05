import numpy as np
import time

N = 5000

def _primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

def construct():
    start_time = time.time()
    deadline = start_time + 290  # 10s safety margin
    
    S = np.arange(N + 1)
    primes = _primes(N)
    
    while True:
        if time.time() > deadline:
            break
            
        L = len(S)
        if L < 2:
            break
            
        # Get relevant primes
        # We can filter primes list each time, or maintain an index
        # Since primes is static, just filter
        relevant_primes = [p for p in primes if p <= L]
        
        violating_prime = None
        min_class_idx = -1
        min_class_count = L + 1
        
        # Check for violations
        # We want to find a prime p where all residue classes are occupied.
        # And among those, we might pick the one that allows us to remove the fewest elements?
        # Or just the smallest p?
        # Let's try picking the p that minimizes the size of the smallest class.
        # This directly minimizes the immediate removal cost.
        
        best_removal_count = L + 1
        best_p = None
        best_r = None
        
        for p in relevant_primes:
            # Count residues
            # S % p creates an array of length L
            # np.bincount is O(L)
            counts = np.bincount(S % p, minlength=p)
            
            # Check if any count is 0
            if counts.min() > 0:
                # It's a violation
                # Find the class with minimum count
                min_idx = np.argmin(counts)
                min_cnt = counts[min_idx]
                
                if min_cnt < best_removal_count:
                    best_removal_count = min_cnt
                    best_p = p
                    best_r = min_idx
                    
        if best_p is None:
            # No violations
            break
        else:
            # Remove elements in class best_r mod best_p
            mask = (S % best_p) != best_r
            S = S[mask]
            
    return S.tolist()
