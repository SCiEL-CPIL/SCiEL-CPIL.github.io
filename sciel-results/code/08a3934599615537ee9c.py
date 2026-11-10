import numpy as np
from math import log
import time

def _compute_mobius(N):
    """Compute Möbius function mu(k) for k=1 to N using a linear sieve."""
    mu = np.ones(N + 1, dtype=np.int8)
    is_prime = np.ones(N + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    primes = []
    for i in range(2, N + 1):
        if is_prime[i]:
            primes.append(i)
            mu[i] = -1
        for p in primes:
            if i * p > N:
                break
            is_prime[i * p] = False
            if i % p == 0:
                mu[i * p] = 0
                break
            else:
                mu[i * p] = -mu[i]
    mu[1] = 1
    return mu

def _compute_M_array(f, N, max_x):
    """Compute M(x) = sum f(k) floor(x/k) for x in [1, max_x] using difference array."""
    diff = np.zeros(max_x + 2, dtype=np.float64)
    for k in range(1, N + 1):
        fk = f[k]
        if fk == 0:
            continue
        j = 1
        while j * k <= max_x:
            start = j * k
            end = min((j + 1) * k - 1, max_x)
            diff[start] += fk * j
            diff[end + 1] -= fk * j
            j += 1
    M = np.cumsum(diff[:max_x + 1])
    return M

def prime_number_theorem_partial_function():
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin before 600s limit
    
    fallback_domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
    fallback_values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
    
    best_score = -np.inf
    best_domain = None
    best_values = None
    
    # Try different N values from large to small
    for N in [200000, 100000, 50000, 20000, 10000, 5000, 2000, 1000]:
        if time.time() > deadline - 30:
            break
        
        mu = _compute_mobius(N)
        f = mu.astype(np.float64)
        
        # Apply normalization: S = sum f(k)/k, f(1) -= S
        k_arr = np.arange(1, N + 1)
        S = np.sum(f[1:] / k_arr)
        f[1] -= S
        
        # Compute M(x) for x in [1, 10N]
        max_x = 10 * N
        M = _compute_M_array(f, N, max_x)
        max_M = np.max(M[1:])  # x from 1 to max_x
        
        if max_M > 1.0001:
            continue
        
        # Compute objective
        a = -np.sum(f[1:] * np.log(k_arr) / k_arr)
        
        if a > best_score:
            best_score = a
            best_domain = k_arr.copy()
            best_values = f[1:].copy()
    
    if best_domain is None:
        return fallback_domain, fallback_values
    
    return best_domain, best_values
