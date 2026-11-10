# EVOLVE-BLOCK-START
import numpy as np
import time


def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    """
    start_time = time.time()
    deadline = start_time + 500  # Safety margin
    
    # Use K=10000 to balance accuracy and computation time
    K = 10000
    
    # Compute Mobius function mu(k) for k=1..K
    mu = np.ones(K + 1, dtype=np.int8)
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
                    
    # Create domain and values (only non-zero mu)
    mask = mu != 0
    domain = np.where(mask)[0] + 1  # 1-indexed
    values = mu[mask].astype(np.float64)
    
    # Check time
    if time.time() > deadline:
         domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
         values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
         return domain, values

    # Verify feasibility
    max_x = 10 * K
    S = np.zeros(max_x + 1, dtype=np.float64)
    
    for k in range(1, K + 1):
        if mu[k] != 0:
            S[k::k] += mu[k]
            
    M = np.cumsum(S)
    
    if np.any(M > 1.0001):
        # Fallback to seed if infeasible
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values

    # Calculate adjustment S_sum = sum f(k)/k
    S_sum = np.sum(values / domain)
    
    # Adjust f(1)
    idx_1 = np.where(domain == 1)[0]
    if len(idx_1) > 0:
        values[idx_1[0]] -= S_sum

    return domain, values

# EVOLVE-BLOCK-END
