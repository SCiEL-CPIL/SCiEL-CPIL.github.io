import math
import time

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # 5s safety margin before 90s kill
    
    N = 1000000
    MAX_N = N + 1  # We need P[N+1]
    
    # Sieve of Eratosthenes for primes up to sqrt(MAX_N)
    limit = int(math.isqrt(MAX_N))
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, limit + 1) if is_prime[i]]
    
    # Initialize powerful part array
    P = [1] * (MAX_N + 1)
    
    # Fill P: for each prime p and each k>=2, multiply P[m] by p^k
    # for all m where p^k exactly divides m (i.e., p^k | m but p^(k+1) does not)
    for p in primes:
        q = p * p
        while q <= MAX_N:
            max_i = MAX_N // q
            for i in range(1, max_i + 1):
                if i % p != 0:
                    P[q * i] *= q
            
            q *= p
            if time.time() - start_time > DEADLINE:
                break
        
        if time.time() - start_time > DEADLINE:
            break
    
    # Find max product
    best_n = 10000
    best_val = 0
    for n in range(10000, N + 1):
        val = P[n] * P[n + 1]
        if val > best_val:
            best_val = val
            best_n = n
            
    return best_n
