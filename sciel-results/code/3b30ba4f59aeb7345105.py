# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def _mobius_sieve(K):
    """Compute the Möbius function mu(n) for n = 1..K using a sieve."""
    mu = np.ones(K + 1, dtype=np.float64)
    is_prime = np.ones(K + 1, dtype=bool)
    primes = []
    is_prime[0] = False
    if K >= 1:
        is_prime[1] = False
    for i in range(2, K + 1):
        if is_prime[i]:
            primes.append(i)
            mu[i] = -1.0
        for p in primes:
            if i * p > K:
                break
            is_prime[i * p] = False
            if i % p == 0:
                mu[i * p] = 0.0
                break
            else:
                mu[i * p] = -mu[i]
    return mu

def _compute_M_max(values, K):
    """
    Compute max_{x in 1..10K} M(x) where M(x) = sum_k f(k) * floor(x/k).
    values is an array of length K+1 (index 0 unused) with f(k) at index k.
    """
    # M(x) = sum_{k=1}^{min(x,K)} f(k) * floor(x/k)
    # For x in 1..10K, we can compute this efficiently.
    # Note: for x > K, floor(x/k) = x/k for k <= x, but we only have f(k) for k <= K.
    # So M(x) = sum_{k=1}^{K} f(k) * floor(x/k) for x >= K.
    
    # We need to check all x in [1, 10K].
    # M(x) is a step function that changes at points where floor(x/k) changes for some k.
    # However, a simpler approach: since K <= 200000, 10K <= 2,000,000.
    # We can compute M(x) for all x in 1..10K using convolution-like methods or direct summation.
    # Direct summation is O(10K * K) which is too slow for K=200000.
    
    # Better approach: M(x) = sum_{k=1}^{K} f(k) * floor(x/k).
    # For a fixed x, floor(x/k) takes O(sqrt(x)) distinct values.
    # But we need max over x.
    
    # Alternative: Note that M(x) = sum_{j=1}^{x} (sum_{k|j, k<=K} f(k)).
    # Let g(j) = sum_{k|j, k<=K} f(k). Then M(x) = sum_{j=1}^{x} g(j).
    # We can compute g(j) for j=1..10K by: for each k, add f(k) to g(j) for all multiples j of k.
    # This is O(K * (10K/k)) = O(10K * H_K) ≈ O(10K log K), which is fine for K=200000.
    
    max_x = 10 * K
    g = np.zeros(max_x + 1, dtype=np.float64)
    for k in range(1, K + 1):
        fk = values[k]
        if fk != 0.0:
            # Add fk to g at positions k, 2k, 3k, ... <= max_x
            g[k::k] += fk
    # M(x) = cumsum of g from 1 to x
    M = np.cumsum(g[1:])
    return np.max(M)

def _compute_metric(values, domain):
    """Compute a = -sum_k f(k) * log(k) / k."""
    # domain is the set of k where f(k) != 0 (or just the domain provided)
    # values corresponds to domain
    if len(domain) == 0:
        return 0.0
    return -np.sum(values * np.log(domain.astype(np.float64)) / domain.astype(np.float64))

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    """
    start_time = time.time()
    deadline = start_time + 620.0  # Safety margin
    
    # Start with a reasonable K
    # The optimum is approached by mu(k) as K -> infinity.
    # a(K) = -sum_{k=1}^K mu(k) log(k)/k -> 1 as K -> infinity.
    # Let's try a few values of K and pick the best feasible one.
    
    best_score = -np.inf
    best_domain = None
    best_values = None
    
    # Try K values: 100, 500, 1000, 5000, 10000, 50000, 100000, 200000
    K_candidates = [100, 500, 1000, 5000, 10000, 50000, 100000, 200000]
    
    for K in K_candidates:
        if time.time() > deadline - 30:
            break
            
        # Compute mu(1..K)
        mu = _mobius_sieve(K)
        
        # f(k) = mu(k) for k=1..K
        # domain = [1, 2, ..., K]
        # values = mu[1..K]
        
        # Check feasibility: M(x) <= 1 for all x in 1..10K
        # For mu, we have the identity sum_{k=1}^{x} mu(k) floor(x/k) = 1 for x >= 1?
        # Actually, sum_{k|n} mu(k) = 1 if n=1, 0 if n>1.
        # So M(x) = sum_{j=1}^{x} sum_{k|j} mu(k) = sum_{j=1}^{x} [j==1] = 1 for all x >= 1.
        # Wait, this is for the full sum. But we are truncating at k <= K.
        # For x <= K, M(x) = sum_{k=1}^{x} mu(k) floor(x/k) = 1 (exactly, since the identity holds for all x).
        # For x > K, M(x) = sum_{k=1}^{K} mu(k) floor(x/k).
        # This is not necessarily 1. It's an approximation.
        # The error is sum_{k=K+1}^{x} mu(k) floor(x/k) (if we had the full sum, it would be 1).
        # Actually, the full sum is 1. So M_truncated(x) = 1 - sum_{k=K+1}^{x} mu(k) floor(x/k).
        # We need M_truncated(x) <= 1.0001. Since the full sum is 1, and we're subtracting a tail,
        # M_truncated(x) = 1 - tail. If tail > 0, M < 1. If tail < 0, M > 1.
        # So we need tail >= -0.0001.
        
        # Let's compute the max M(x) to check feasibility.
        # For efficiency, we can compute M(x) for x in 1..10K.
        
        values_full = mu[1:K+1]
        
        # Compute max M(x)
        max_x = 10 * K
        g = np.zeros(max_x + 1, dtype=np.float64)
        for k in range(1, K + 1):
            fk = mu[k]
            if fk != 0.0:
                g[k::k] += fk
        M = np.cumsum(g[1:])
        max_M = np.max(M)
        
        if max_M > 1.0001:
            continue
        
        # Compute metric
        domain = np.arange(1, K + 1, dtype=np.int64)
        a = -np.sum(values_full * np.log(domain.astype(np.float64)) / domain.astype(np.float64))
        
        if a > best_score:
            best_score = a
            best_domain = domain
            best_values = values_full.copy()
    
    if best_domain is None:
        # Fallback to the original seed
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values
    
    return best_domain, best_values

# EVOLVE-BLOCK-END
