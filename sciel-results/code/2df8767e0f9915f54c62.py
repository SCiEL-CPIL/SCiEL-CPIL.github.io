import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s limit
    N = 1000000
    
    # Sieve of Eratosthenes to get primes up to sqrt(N)
    sqrt_N = int(math.isqrt(N))
    is_prime = bytearray(b'\x01') * (sqrt_N + 1)
    is_prime[0] = 0
    is_prime[1] = 0
    for i in range(2, int(math.isqrt(sqrt_N)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    
    primes = [i for i in range(2, sqrt_N + 1) if is_prime[i]]
    
    # Compute powerful part P(n) for all n in [1, N] using multiplicative sieve
    P = [1] * (N + 1)
    
    for p in primes:
        p2 = p * p
        if p2 > N:
            break
        
        pk = p2  # p^2
        k = 2
        while pk <= N:
            limit = N // pk
            # Iterate j from 1 to limit, applying pk to P[j*pk]
            # only when j % p != 0. Skip every p-th j by block stepping.
            j = 1
            while j <= limit:
                # Apply to j, j+1, ..., j+p-2 (p-1 consecutive j values)
                end = min(j + p - 1, limit + 1)
                for jj in range(j, end):
                    P[jj * pk] *= pk
                j += p
            
            pk *= p
            k += 1
        
        if time.time() > deadline - 5:
            break
    
    # Find max P[n] * P[n+1] for n in [10000, N-1]
    best_val = 0
    best_n = 10000
    for n in range(10000, N):
        val = P[n] * P[n + 1]
        if val > best_val:
            best_val = val
            best_n = n
    
    return best_n
