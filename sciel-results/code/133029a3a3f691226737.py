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
            
        q = p * p
        while q <= MAXN:
            next_q = q * p
            
            # Multiply all multiples of q by q
            pp[q::q] *= q
            
            # Divide all multiples of next_q by q
            # This effectively removes the contribution of q for numbers
            # that have a higher power of p
            if next_q <= MAXN:
                pp[next_q::next_q] /= q
            
            q = next_q
    
    if elapsed() > DEADLINE:
        return 10000
    
    # Scan for the best pair
    start = 10000
    # n ranges from start to MAXN-1
    # pp[n] and pp[n+1]
    # products = pp[start:MAXN] * pp[start+1:MAXN+1]
    
    products = pp[start:MAXN] * pp[start+1:MAXN+1]
    
    max_idx = np.argmax(products)
    best_n = start + max_idx
    
    return int(best_n)
