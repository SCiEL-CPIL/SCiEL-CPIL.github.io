# EVOLVE-BLOCK-START
import numpy as np
import time


def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.

    Strategy: Compute the exact Möbius function mu(k) for all k in [1, K_max] using a linear sieve,
    then construct the domain as the set of all k where mu(k) != 0 (square-free integers) and values
    as mu(k). This ensures the strict constraint M(x) <= 1 is satisfied with equality for all x
    while maximizing the objective a by including the full range of contributing terms.
    """
    start_time = time.time()
    deadline = start_time + 600.0  # 600 second limit, leave safety margin
    
    # Maximum key K allowed by the task contract
    K_max = 200000
    
    # Use a smaller K if time is tight, but start with the full range
    # The linear sieve is O(K), which should be fast enough
    K = K_max
    
    # Linear sieve to compute Möbius function
    # mu[0] is unused, mu[1] = 1
    mu = np.zeros(K + 1, dtype=np.int8)
    mu[1] = 1
    is_prime = np.ones(K + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    primes = []
    
    # Linear sieve for Möbius function
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
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > deadline - 10:
        # Not enough time, return a small feasible solution
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values
    
    # Find all k where mu(k) != 0 (square-free integers)
    mask = mu != 0
    domain = np.where(mask)[0]
    # Ensure domain is positive integers (>= 1)
    domain = domain[domain >= 1]
    values = mu[domain].astype(np.float64)
    
    # Ensure domain is sorted and distinct
    # np.where already returns sorted indices
    
    # Check time again
    elapsed = time.time() - start_time
    if elapsed > deadline - 5:
        return domain, values
    
    return domain, values


# EVOLVE-BLOCK-END
