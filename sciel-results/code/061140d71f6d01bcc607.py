import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin

    N = 1000000
    max_val = N + 1

    # Sieve of Eratosthenes to get primes up to sqrt(N+1)
    limit = int(math.isqrt(max_val))
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    
    primes = [i for i in range(2, limit + 1) if is_prime[i]]
    
    # Use a Python list for faster element-wise updates in the loop
    pp = [1] * (max_val + 2)
    
    for p in primes:
        p2 = p * p
        if p2 > max_val:
            continue
        
        max_k = max_val // p2
        
        # Precompute v_p(k) for k in [1, max_k]
        # v[k] = exponent of p in k
        v = [0] * (max_k + 1)
        # v[k] = v[k//p] + 1 if k % p == 0 else 0
        # We can fill this in O(max_k)
        for k in range(p, max_k + 1, p):
            v[k] = v[k // p] + 1
            
        # Update pp
        # m = k * p2
        # pp[m] *= p^(2 + v[k])
        # Precompute powers of p to avoid repeated exponentiation
        # Max exponent is 2 + log_p(max_k)
        # log_2(10^6) is ~20. So max exp ~22.
        # We can just use p ** (2 + v[k])
        
        for k in range(1, max_k + 1):
            m = k * p2
            exp = 2 + v[k]
            pp[m] *= p ** exp
            
        # Check deadline
        if time.time() > deadline:
            break
    
    # Convert to Numpy for fast vectorized multiplication
    pp_np = np.array(pp, dtype=np.int64)
    
    # Find max product pp[n] * pp[n+1] for n in [10000, 1000000]
    products = pp_np[10000:N+1] * pp_np[10001:N+2]
    idx = np.argmax(products)
    best_n = 10000 + idx
    
    return int(best_n)
