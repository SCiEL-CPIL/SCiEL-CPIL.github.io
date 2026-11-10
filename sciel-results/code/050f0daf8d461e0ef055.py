import math

def construct():
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
    
    # Fill P
    for p in primes:
        q = p * p
        # q is p^k, starting with k=2
        while q <= MAX_N:
            # For all m = q * i such that i is not divisible by p
            # m <= MAX_N => i <= MAX_N // q
            max_i = MAX_N // q
            # Iterate i from 1 to max_i
            # We want to skip i divisible by p
            # We can do this by iterating i and checking, or stepping carefully.
            # Checking is O(1) and fast enough.
            for i in range(1, max_i + 1):
                if i % p != 0:
                    P[q * i] *= q
            q *= p
            
    # Find max product
    best_n = 10000
    best_val = 0
    # n in [10000, 1000000]
    # We need P[n] and P[n+1]
    for n in range(10000, N + 1):
        val = P[n] * P[n + 1]
        if val > best_val:
            best_val = val
            best_n = n
            
    return best_n
