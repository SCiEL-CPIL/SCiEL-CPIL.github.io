# EVOLVE-BLOCK-START
import numpy as np
import time

def prime_number_theorem_partial_function():
    start_time = time.time()
    DEADLINE = 600.0  # safety margin
    
    # Compute Mobius function up to K using sieve
    K = 200000
    mu = np.ones(K + 1, dtype=np.float64)
    is_prime = np.ones(K + 1, dtype=bool)
    
    # Sieve of Eratosthenes + Mobius
    for i in range(2, K + 1):
        if is_prime[i]:
            for j in range(i, K + 1, i):
                is_prime[j] = False if j != i else is_prime[j]
                mu[j] *= -1
            # Mark i*i multiples
            i2 = i * i
            if i2 <= K:
                for j in range(i2, K + 1, i2):
                    mu[j] = 0.0
    # Fix: is_prime should mark primes; let me redo properly
    # Actually let me use a cleaner sieve
    
    # Redo: standard Mobius sieve
    mu = np.ones(K + 1, dtype=np.float64)
    mu[0] = 0
    primes = []
    is_composite = np.zeros(K + 1, dtype=bool)
    
    for i in range(2, K + 1):
        if not is_composite[i]:
            primes.append(i)
        for p in primes:
            if i * p > K:
                break
            is_composite[i * p] = True
            if i % p == 0:
                mu[i * p] = 0.0
                break
            else:
                mu[i * p] = -mu[i]
    
    # mu[1] = 1 by default
    mu[1] = 1.0
    
    # Build domain and values for all k where mu[k] != 0
    domain_list = []
    values_list = []
    for k in range(1, K + 1):
        if mu[k] != 0:
            domain_list.append(k)
            values_list.append(mu[k])
    
    domain = np.array(domain_list, dtype=np.int64)
    values = np.array(values_list, dtype=np.float64)
    
    # Verify feasibility: check M(x) for a sample of x values
    # M(x) = sum_k f(k) * floor(x/k)
    # For mu, M(x) = sum_{k<=x} mu(k) floor(x/k) = 1 for all x >= 1
    # Let's verify for a few values
    max_K = int(domain.max()) if len(domain) > 0 else 1
    check_points = [1, 2, 5, 10, 100, 1000, max_K, 2*max_K, 5*max_K, 10*max_K]
    check_points = [min(x, 10*max_K) for x in check_points if 1 <= x <= 10*max_K]
    
    # For efficiency, compute M(x) for check points
    # M(x) = sum_k f(k) * floor(x/k)
    for x in check_points:
        # vectorized
        floors = np.floor(x / domain).astype(np.int64)
        Mx = np.sum(values * floors)
        if Mx > 1.0001:
            # Infeasible, fall back to seed
            domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
            values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
            return domain, values
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - 30:
        # Return what we have
        pass
    
    return domain, values

# EVOLVE-BLOCK-END
