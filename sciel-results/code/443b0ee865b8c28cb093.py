import numpy as np

def prime_number_theorem_partial_function():
    K = 20000
    # Compute Mobius function using sieve
    mu = np.ones(K + 1, dtype=np.int64)
    is_prime = np.ones(K + 1, dtype=bool)
    primes = []
    mu[0] = 0
    is_prime[0] = False
    is_prime[1] = False
    
    # Standard linear sieve for Mobius
    # Actually, a simple sieve is fine for 20k
    mu = np.ones(K + 1)
    mu[0] = 0
    is_prime = np.ones(K + 1, dtype=bool)
    primes = []
    
    for i in range(2, K + 1):
        if is_prime[i]:
            primes.append(i)
            mu[i] = -1
        for p in primes:
            if i * p > K:
                break
            is_prime[i * p] = False
            if i % p == 0:
                mu[i * p] = 0
                break
            else:
                mu[i * p] = -mu[i]
                
    # Filter domain
    domain = np.array([k for k in range(1, K + 1) if mu[k] != 0], dtype=np.int64)
    values = np.array([mu[k] for k in domain], dtype=np.float64)
    
    return domain, values
