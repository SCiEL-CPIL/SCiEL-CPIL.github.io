import time
import numpy as np

def construct():
    t0 = time.time()
    DEADLINE = 85  # safety margin before 90s kill

    def elapsed():
        return time.time() - t0

    MAXN = 1_000_000
    
    # Initialize powerful part array with 1s
    pp = np.ones(MAXN + 2, dtype=np.int64)
    
    # Sieve for primes up to MAXN
    is_prime = np.ones(MAXN + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(MAXN**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    primes = np.where(is_prime)[0]
    
    # Calculate powerful parts
    for p in primes:
        if p * p > MAXN:
            break
        if elapsed() > DEADLINE:
            break
            
        p_pow = p * p
        j = 2
        while p_pow <= MAXN:
            if elapsed() > DEADLINE:
                break
            
            # We want to multiply pp[m] by p^j for all m such that v_p(m) = j
            # m = p^j * k where k is not divisible by p
            # k takes values 1, 2, ..., p-1, p+1, p+2, ...
            # This corresponds to p-1 arithmetic progressions with step p^(j+1)
            # For k = 1 to p-1:
            #   start = p^j * k
            #   step = p^(j+1)
            
            step = p_pow * p
            for k in range(1, p):
                start = p_pow * k
                # Apply slice
                pp[start::step] *= p_pow
            
            p_pow *= p
            j += 1

    if elapsed() > DEADLINE:
        return 10000
    
    # Scan for the best pair
    start = 10000
    # n ranges from start to MAXN-1
    # We need pp[n] * pp[n+1]
    # pp is 1-indexed in logic? No, pp[0] is unused effectively, pp[1] corresponds to 1.
    # The array `pp` has size MAXN+2.
    # pp[i] is the powerful part of i.
    
    products = pp[start:MAXN] * pp[start+1:MAXN+1]
    
    max_idx = np.argmax(products)
    best_val = products[max_idx]
    best_n = start + max_idx
    
    return int(best_n)
