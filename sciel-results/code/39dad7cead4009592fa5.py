# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_mobius(K):
    """Compute Mobius function mu(n) for n = 1..K using a linear sieve."""
    mu = np.ones(K + 1, dtype=np.int8)
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
    return mu


def _compute_M_max(domain, values, max_x):
    """
    Compute max of M(x) = sum_k f(k) * floor(x/k) for x in [1, max_x].
    Uses g(n) = sum_{k|n} f(k), then M(x) = cumsum(g).
    """
    g = np.zeros(max_x + 1, dtype=np.float64)
    for k, f_k in zip(domain, values):
        if f_k == 0.0:
            continue
        g[k::k] += f_k
    M = np.cumsum(g)
    return np.max(M)


def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    """
    start_time = time.time()
    DEADLINE = 570  # safety margin before 630s kill

    # Use the full Mobius function up to K = 200000
    K = 200000
    mu = _compute_mobius(K)

    # Retain only square-free indices (mu(k) != 0)
    mask = mu != 0
    domain = np.where(mask)[0]
    domain = domain[domain >= 1]
    values = mu[domain].astype(np.float64)

    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 30:
        # Verify feasibility: M(x) <= 1.0001 for all x in [1, 10*K]
        max_x = 10 * K
        max_M = _compute_M_max(domain, values, max_x)
        
        if max_M > 1.0001:
            # Fallback: try smaller K
            for K_try in [100000, 50000, 20000, 10000]:
                mu_try = _compute_mobius(K_try)
                mask_try = mu_try != 0
                domain_try = np.where(mask_try)[0]
                domain_try = domain_try[domain_try >= 1]
                values_try = mu_try[domain_try].astype(np.float64)
                
                max_x_try = 10 * K_try
                max_M_try = _compute_M_max(domain_try, values_try, max_x_try)
                if max_M_try <= 1.0001:
                    domain = domain_try
                    values = values_try
                    break
                if time.time() - start_time > DEADLINE - 10:
                    break

    return domain, values


# EVOLVE-BLOCK-END
