# EVOLVE-BLOCK-START
import numpy as np
import time


def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    """
    start_time = time.time()
    deadline = start_time + 550  # safety margin before 630s kill
    
    # Compute Möbius function up to K
    K = 2000
    mu = np.ones(K + 1, dtype=np.float64)
    mu[0] = 0
    # Sieve for Möbius
    is_prime = np.ones(K + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    primes = []
    for i in range(2, K + 1):
        if is_prime[i]:
            primes.append(i)
            mu[i] = -1
            for j in range(i * i, K + 1, i):
                if is_prime[j]:
                    is_prime[j] = False
                mu[j] *= -1
            # Handle prime squares
            for j in range(i * i, K + 1, i * i):
                mu[j] = 0
    
    # Actually, let me use a proper Möbius sieve
    mu = np.ones(K + 1, dtype=np.float64)
    mu[0] = 0
    is_prime = np.ones(K + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, K + 1):
        if is_prime[i]:
            for j in range(i, K + 1, i):
                is_prime[j] = False
                mu[j] *= -1
            for j in range(i * i, K + 1, i * i):
                mu[j] = 0
    
    # Get non-zero mu values
    domain = np.array([i for i in range(1, K + 1) if mu[i] != 0], dtype=np.int64)
    values = np.array([mu[i] for i in range(1, K + 1) if mu[i] != 0], dtype=np.float64)
    
    m = len(domain)
    K_max = domain[-1]
    
    # Check M(x) = sum_k f(k) * floor(x/k) for x in [1, 10*K_max]
    # This is expensive: m * 10*K_max operations
    # For K=2000, m~1200, 10*K_max=20000, cost ~ 2.4*10^7 - feasible
    
    # But wait, we need to check ALL x in [1, 10*K_max]. That's 20000 values of x.
    # For each x, M(x) = sum_{k in domain} f(k) * floor(x/k)
    # Total: 20000 * 1200 = 2.4*10^7 - OK
    
    # Actually, let's be smarter. M(x) changes only at multiples of domain elements.
    # But checking all x is simpler and fast enough for K=2000.
    
    max_x = 10 * K_max
    # Compute M(x) for all x in [1, max_x]
    # M(x) = sum_{k} f(k) * floor(x/k)
    # We can compute this incrementally
    
    M = np.zeros(max_x + 1, dtype=np.float64)
    for k, f in zip(domain, values):
        M[k:] += f * (np.arange(k, max_x + 1) // k)
    
    if np.max(M[1:]) > 1.0001:
        # Infeasible, fall back to seed
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values
    
    return domain, values


# EVOLVE-BLOCK-END
