# EVOLVE-BLOCK-START
import numpy as np
import time

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.

    Strategy:
      The objective is a = -sum_k f(k) log(k)/k, subject to:
        1) After clipping to [-10,10] and adjusting f(1) so sum_k f(k)/k = 0,
           M(x) = sum_k f(k) floor(x/k) <= 1.0001 for all x in [1, 10K].
        2) K <= 200000, m >= 1.

      We use the Mobius function mu(k) for k=1..K.
      For mu, M(x) = 1 for all x >= 1 (exact identity for the infinite sum, and 
      the partial sum up to K is 1 for x <= K, and close to 1 for x > K up to 10K 
      because the tail of the sum is small).
      
      Verification is done in O(K log K) time using a difference array for the 
      divisor sum g(j) = sum_{k|j} f(k), then prefix sum for M(x).
    """
    start_time = time.time()
    time_limit = 630.0
    safety_margin = 10.0

    # Maximum allowed K
    K = 200000

    # 1. Compute Mobius function mu(k) for k=1..K using a linear sieve
    mu = np.zeros(K + 1, dtype=np.int64)
    mu[1] = 1
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

    # Domain is 1..K
    domain = np.arange(1, K + 1, dtype=np.int64)
    # Values are mu(k)
    values = mu[1:].astype(np.float64)

    # 2. Clip values to [-10, 10]
    # mu(k) is in {-1, 0, 1}, so clipping has no effect, but we do it for correctness
    values_clipped = np.clip(values, -10.0, 10.0)

    # 3. Adjust f(1) to satisfy sum f(k)/k = 0
    S = np.sum(values_clipped / domain)
    values_adj = values_clipped.copy()
    values_adj[0] -= S

    # 4. Verify M(x) <= 1.0001 for x in [1, 10K]
    x_max = 10 * K
    # We need to compute M(x) = sum_{j=1}^x g(j) where g(j) = sum_{k|j} f(k)
    # g(j) can be computed by iterating k and adding f(k) to all multiples of k.
    # Let diff be an array of size x_max + 1.
    # For each k, diff[k], diff[2k], ..., diff[floor(x_max/k)*k] += f(k)
    # Then g = diff, and M = cumsum(g).
    
    # Check time before starting the heavy loop
    if time.time() - start_time > time_limit - safety_margin:
        domain_small = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values_small = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain_small, values_small

    diff = np.zeros(x_max + 1, dtype=np.float64)
    
    # Vectorized update for each k
    for k in range(1, K + 1):
        f_k = values_adj[k - 1]
        if f_k == 0:
            continue
        # Add f_k to indices k, 2k, ..., <= x_max
        # Slice: start=k, stop=x_max+1, step=k
        diff[k:x_max+1:k] += f_k
        
        # Time check every 10000 iterations
        if k % 10000 == 0:
            if time.time() - start_time > time_limit - safety_margin:
                domain_small = np.array([1, 2, 3, 5, 30], dtype=np.int64)
                values_small = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
                return domain_small, values_small

    # Compute M(x) = cumsum(diff)
    # M[0] is 0. We care about x in [1, x_max].
    M = np.cumsum(diff)
    
    # Check max M(x) for x in [1, x_max]
    max_M = np.max(M[1:x_max+1])
    
    if max_M > 1.0001:
        # Infeasible. Fall back to a small known-good solution.
        # This should not happen for Mobius, but just in case.
        domain_small = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values_small = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain_small, values_small

    # 5. Compute metric a = -sum_k f(k) log(k) / k
    log_k = np.log(domain)
    a = -np.sum(values_adj * log_k / domain)

    return domain, values

# EVOLVE-BLOCK-END
