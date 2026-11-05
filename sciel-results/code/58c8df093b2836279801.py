import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    N = 1000000
    START = 10000
    
    # Sieve for primes
    is_prime = np.ones(N + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(np.sqrt(N)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    primes = np.where(is_prime)[0]
    
    # Powerful part array
    P = np.ones(N + 1, dtype=np.int64)
    
    for p in primes:
        p = int(p)
        power = p * p
        if power > N:
            continue
            
        # First step: multiply by p^2
        P[power::power] *= power
        
        power *= p
        # Subsequent steps: multiply by p
        while power <= N:
            P[power::power] *= p
            power *= p
            
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    # Calculate products
    # n in [START, N-1]
    # P[n] * P[n+1]
    products = P[START:N] * P[START+1:N+1]
    
    if time.time() > deadline:
        # Fallback
        return START
        
    max_idx = np.argmax(products)
    best_n = START + int(max_idx)
    
    return best_n
