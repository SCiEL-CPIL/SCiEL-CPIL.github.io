import time
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 75  # safety margin before 90s kill
    
    MAX_N = 1000001  # we need up to 1000000, and n+1 can be 1000001
    
    # Sieve to find primes up to MAX_N
    is_prime = np.ones(MAX_N + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(MAX_N**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    primes = np.where(is_prime)[0]
    
    # Initialize powerful_part array
    pp = np.ones(MAX_N + 1, dtype=np.int64)
    
    # For each prime, accumulate the full prime power into multiples
    for p in primes:
        if p * p > MAX_N:
            break
        p2 = p * p
        # Find the highest power p^e <= MAX_N
        powers = []
        pe = p2
        while pe <= MAX_N:
            powers.append(pe)
            pe_next = pe * p
            if pe_next > MAX_N:
                break
            pe = pe_next
        
        # Process from highest power to lowest
        # For p^e: multiples of p^e but NOT p^(e+1) get exactly p^e
        # We process in order: first assign p^2 to all multiples of p^2
        # Then for multiples of p^3, we need to "upgrade" from p^2 to p^3
        # Easier: for each power level, handle the numbers that have EXACTLY that exponent
        
        # Alternative: for each multiple of p^2, compute the exact exponent
        # But that's slow. Let's use the layered approach:
        # Start with p^2 for all multiples of p^2
        # Then for multiples of p^3, multiply by p (to go from p^2 to p^3)
        # Then for multiples of p^4, multiply by p (to go from p^3 to p^4)
        # etc.
        
        # This is cleaner: 
        # pp[multiples of p^2] *= p^2
        # pp[multiples of p^3] *= p  (now they have p^3)
        # pp[multiples of p^4] *= p  (now they have p^4)
        # etc.
        
        # Apply p^2 to all multiples of p^2
        multiples_p2 = np.arange(p2, MAX_N + 1, p2)
        pp[multiples_p2] *= p2
        
        # Apply additional p factors for higher powers
        pe = p2 * p  # p^3
        while pe <= MAX_N:
            multiples_pe = np.arange(pe, MAX_N + 1, pe)
            pp[multiples_pe] *= p
            pe *= p
        
        if time.time() - start_time > DEADLINE:
            break
    
    # Now compute the product pp[n] * pp[n+1] for n in [10000, 1000000]
    # and find the n that maximizes it
    products = pp[10000:1000001] * pp[10001:1000002]
    
    # Find the index of maximum
    max_idx = np.argmax(products)
    best_n = 10000 + max_idx
    best_val = products[max_idx]
    
    # Verify
    if best_n < 10000 or best_n > 1000000:
        return 10000
    
    return int(best_n)
