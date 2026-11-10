# EVOLVE-BLOCK-START
import numpy as np
import time


def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes the evaluator's objective a = -sum_k f(k)*log(k)/k subject to:
      - f values clipped to [-10,10]
      - after adjusting f(1) so that sum_k f(k)/k == 0
      - M(x) = sum_k f(k) * floor(x/k) <= 1.0001 for all integer x in [1, 10*K]
    where K is the maximum domain key (<= 200000).

    Strategy: Use the Möbius function mu(k) as f(k) for k=1..K. The classic identity
    sum_{k=1}^{x} mu(k)*floor(x/k) = 1 holds for all x >= 1, making M(x)=1 feasible.
    The objective a = -sum_{k=1}^{K} mu(k)*log(k)/k approaches 1 as K -> infinity.

    To maximize score within time limits, we choose the largest feasible K such that
    the evaluator's O(m * 10K) computation fits in the 630s wall-clock limit.
    """
    t_start = time.time()
    deadline = 600.0  # safety margin before 630s hard limit
    
    # We need to pick K such that the evaluator can compute M(x) for x in [1, 10K]
    # with m non-zero entries in O(m * 10K) time.
    # For mu, m ~ 6K/pi^2 * K = 0.6079 * K entries are non-zero.
    # So ops ~ 0.6079 * K * 10 * K = 6.079 * K^2.
    # At ~1e8 ops/sec in numpy (conservative), time ~ 6.079e-8 * K^2 seconds.
    # For 300s: K^2 < 300/6.079e-8 = 4.93e9, K < ~70000.
    # Let's be conservative and use K = 50000.
    # m ~ 30396, ops ~ 30396 * 500000 = 1.52e10. At 1e8 ops/s that's 152s. OK.
    # Actually numpy vectorized ops are much faster. Let's try K = 100000.
    # m ~ 60793, ops ~ 60793 * 1000000 = 6.08e10. Could be ~600s+. Risky.
    # Let's go with K = 80000 as a balance.
    
    K = 80000
    
    # Check if we have enough time to compute mu for K
    # Computing mu via sieve is O(K log log K), very fast.
    
    # Compute Möbius function up to K
    mu = np.ones(K + 1, dtype=np.int8)
    mu[0] = 0
    
    # Sieve for primes
    is_prime = np.ones(K + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(np.sqrt(K)) + 1):
        if is_prime[i]:
            is_prime[i * i::i] = False
    
    primes = np.where(is_prime)[0]
    
    for p in primes:
        mu[p::p] = -mu[p::p]
        p2 = p * p
        if p2 <= K:
            mu[p2::p2] = 0
    
    # Only include k where mu(k) != 0 to reduce m
    non_zero = mu[1:] != 0
    domain = np.where(non_zero)[0] + 1  # shift to 1-based
    values = mu[domain].astype(np.float64)
    
    elapsed = time.time() - t_start
    # If computation took too long, fall back to small solution
    if elapsed > 60:
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values
    
    return domain, values


# EVOLVE-BLOCK-END
