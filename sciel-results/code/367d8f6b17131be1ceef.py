# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_mobius(N):
    """Compute Moebius function mu(k) for k = 1..N using linear sieve."""
    mu = np.ones(N + 1, dtype=np.int8)
    mu[0] = 0
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


def _verify_feasibility(domain, values, K, max_x=100000):
    """
    Check M(x) = sum_k f(k) * floor(x/k) <= 1.0001 for x in [1, max_x].
    Uses the fact that floor(x/k) changes only at multiples of k.
    We check at all critical points: x = 1, 2, ..., max_x but efficiently.
    """
    m = len(domain)
    if m == 0:
        return True
    
    # M(x) = sum_k f(k) * floor(x/k)
    # For a given x, floor(x/k) is constant between multiples.
    # The maximum of M(x) occurs at x = 1, 2, ..., max_x.
    # But we can be smarter: M(x) is a step function that changes at x = k, 2k, 3k, ...
    # Let's just compute M(x) for all x in [1, max_x] using a difference array approach.
    
    # M(x) = sum_k f(k) * floor(x/k)
    # floor(x/k) = number of multiples of k in [1, x]
    # So M(x) = sum_k f(k) * (number of j such that j*k <= x)
    #          = sum_{j>=1} sum_k f(k) * [j*k <= x]
    #          = sum_{n<=x} sum_{k|n} f(k)
    
    # Let g(n) = sum_{k|n} f(k). Then M(x) = sum_{n=1}^{x} g(n).
    # We need sum_{n=1}^{x} g(n) <= 1.0001 for all x in [1, max_x].
    
    # Compute g(n) for n up to max_x using a sieve-like approach.
    g = np.zeros(max_x + 1, dtype=np.float64)
    for i in range(m):
        k = domain[i]
        v = values[i]
        # Add v to g(n) for all n that are multiples of k
        for n in range(k, max_x + 1, k):
            g[n] += v
    
    # Now M(x) = cumsum(g)[x]
    M = np.cumsum(g)
    # Check M(x) <= 1.0001 for x in [1, max_x]
    if np.any(M[1:max_x+1] > 1.0001):
        return False
    return True


def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.

    Returns:
        (domain, values): a tuple of two numpy arrays of equal length describing f
    """
    start_time = time.time()
    deadline = 600.0
    
    # Strategy: Use the Mobius function mu(k) for k = 1..K.
    # For the full Mobius function, M(x) = 1 for all x >= 1 (since sum_{d|n} mu(d) = 0 for n>1).
    # For a truncated version (k <= K), M(x) = 1 for x <= K, but for x > K, 
    # M(x) = sum_{k<=K} mu(k) * floor(x/k) which may deviate from 1.
    # 
    # The objective a = -sum_{k=1}^{K} mu(k) * log(k) / k approaches 1 as K -> infinity.
    # We want to find the largest K such that M(x) <= 1.0001 for all x in [1, 10K].
    
    # Start with a reasonable K and increase.
    # The constraint is that for x in [K+1, 10K], M(x) = sum_{k<=K} mu(k) floor(x/k) must be <= 1.0001.
    
    # Let's try increasing K values and check feasibility.
    # For efficiency, we'll test specific K values.
    
    best_domain = None
    best_values = None
    best_a = -np.inf
    
    # Try a range of K values
    # For K up to ~5000, the computation is fast
    # For larger K, we need to be careful about the 10K range
    
    K_candidates = []
    # Start with small values and increase
    for K in [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000]:
        K_candidates.append(K)
    
    for K in K_candidates:
        if time.time() > deadline - 60:
            break
            
        max_x = 10 * K
        if max_x > 2000000:  # Too large to compute efficiently
            break
        
        mu = _compute_mobius(K)
        
        # domain: k where mu(k) != 0 (or include all, but mu(k)=0 contributes nothing)
        # Actually, let's include all k=1..K for clarity, but only non-zero mu matter
        non_zero = np.where(mu[1:K+1] != 0)[0] + 1
        if len(non_zero) == 0:
            continue
        
        domain = non_zero.astype(np.int64)
        values = mu[domain].astype(np.float64)
        
        # Verify feasibility
        if not _verify_feasibility(domain, values, K, max_x):
            continue
        
        # Compute objective a = -sum_k f(k) * log(k) / k
        log_k = np.log(domain.astype(np.float64))
        a = -np.sum(values * log_k / domain)
        
        if a > best_a:
            best_a = a
            best_domain = domain.copy()
            best_values = values.copy()
        
        # If a is very close to 1, we can stop
        if best_a > 0.999:
            break
    
    # Fallback: use the seed program's values if nothing worked
    if best_domain is None:
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
    else:
        domain = best_domain
        values = best_values
    
    return domain, values


# EVOLVE-BLOCK-END
