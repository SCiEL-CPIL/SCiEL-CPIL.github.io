import time

def _sieve_primes(limit):
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    return [i for i, prime in enumerate(is_prime) if prime]

def _phi(n, primes):
    if n == 1:
        return 1
    result = n
    temp = n
    for p in primes:
        if p * p > temp:
            break
        if temp % p == 0:
            result -= result // p
            while temp % p == 0:
                temp //= p
    if temp > 1:
        result -= result // temp
    return result

def construct():
    start = time.time()
    deadline = start + 85
    
    # Primes up to 2000 is sufficient for factoring n up to 4,000,000
    # We only need up to 1000 for n up to 1,000,000, but 2000 is cheap.
    primes = _sieve_primes(2000)
    
    for n in range(1, 1000001):
        if time.time() > deadline:
            return 14400 # Fallback
        phi_n = _phi(n, primes)
        if phi_n % 120 == 0:
            return n
            
    return 14400
