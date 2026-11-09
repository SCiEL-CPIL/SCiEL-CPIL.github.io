import numpy as np
import time

def compute_mobius(n):
    mu = np.ones(n + 1, dtype=np.int64)
    mu[0] = 0
    is_prime = np.ones(n + 1, dtype=bool)
    primes = []
    for i in range(2, n + 1):
        if is_prime[i]:
            primes.append(i)
            mu[i] = -1
        for p in primes:
            if i * p > n:
                break
            is_prime[i * p] = False
            if i % p == 0:
                mu[i * p] = 0
                break
            else:
                mu[i * p] = -mu[i]
    return mu

def check_feasibility(mu, K):
    """Check if M(x) <= 1.0001 for all x in [1, 10K]."""
    max_x = 10 * K
    # For each k with mu[k] != 0, add mu[k]*j to positions [j*k, (j+1)*k - 1]
    # Using difference array: diff[j*k] += mu[k]*j, diff[(j+1)*k] -= mu[k]*j
    
    diff = np.zeros(max_x + 2, dtype=np.float64)
    
    nonzero_k = np.nonzero(mu[1:K+1])[0] + 1
    mu_vals = mu[1:K+1][nonzero_k - 1].astype(np.float64)
    
    for k, mk in zip(nonzero_k, mu_vals):
        if mk == 0:
            continue
        j_max = max_x // k
        if j_max == 0:
            continue
        j = np.arange(1, j_max + 1, dtype=np.int64)
        starts = j * k
        ends = (j + 1) * k
        # Use np.add.at for repeated indices
        np.add.at(diff, starts, mk * j)
        np.add.at(diff, ends, -mk * j)
    
    M = np.cumsum(diff)
    M = M[1:max_x + 1]
    return np.max(M) <= 1.0001, M

def prime_number_theorem_partial_function():
    start_time = time.time()
    deadline = start_time + 580
    
    max_K = 200000
    mu = compute_mobius(max_K)
    
    for K in [200000, 150000, 100000, 75000, 50000, 25000, 10000, 5000]:
        if time.time() > deadline - 10:
            break
        feasible, M = check_feasibility(mu, K)
        if feasible:
            mask = mu[1:K + 1] != 0
            domain = np.nonzero(mask)[0] + 1
            values = mu[1:K + 1][mask].astype(np.float64)
            return domain, values
    
    # Fallback
    domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
    values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
    return domain, values
