# EVOLVE-BLOCK-START
import numpy as np
import time

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    """
    start_time = time.time()
    
    # Deadline guard: 600s limit, keep 30s margin
    deadline = start_time + 570
    
    # Generate Mobius function up to a limit
    # We need square-free k with mu(k) = -1
    # Let's go up to K=10000 to have enough candidates
    K_limit = 10000
    
    # Sieve for Mobius
    mu = np.ones(K_limit + 1, dtype=np.int8)
    is_prime = np.ones(K_limit + 1, dtype=bool)
    primes = []
    for i in range(2, K_limit + 1):
        if is_prime[i]:
            primes.append(i)
            mu[i] = -1
            for j in range(i, K_limit + 1, i):
                is_prime[j] = False
                mu[j] *= -1
                if (j // i) % i == 0:
                    mu[j] = 0

    # Candidates: k > 1, mu(k) = -1
    candidates = [k for k in range(2, K_limit + 1) if mu[k] == -1]
    
    # Sort by k descending (largest log k first)
    candidates.sort(reverse=True)
    
    S = []
    current_sum_inv = 0.0
    limit_sum_inv = 1.0 # Must be <= 1.0 to keep M(1) <= 1
    
    for k in candidates:
        inv_k = 1.0 / k
        if current_sum_inv + inv_k <= limit_sum_inv:
            S.append(k)
            current_sum_inv += inv_k
            
        if time.time() > deadline:
            break

    # If we didn't find any, use a fallback
    if not S:
        S = [2]
        current_sum_inv = 0.5

    # Construct domain and values
    domain = np.array([1] + S, dtype=np.int64)
    # f(1) is arbitrary, will be replaced
    # f(k) = -1 for k in S
    values = np.zeros(len(domain), dtype=np.float64)
    values[0] = 0.0 
    values[1:] = -1.0

    return domain, values

# EVOLVE-BLOCK-END
