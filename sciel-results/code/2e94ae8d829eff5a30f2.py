# EVOLVE-BLOCK-START
import numpy as np
import time

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.

    Strategy:
      1. Use the Mobius function mu(k) for k=1..K.
      2. K = 200,000 (max allowed).
      3. Compute mu(k) via linear sieve.
      4. Clip values to [-10, 10].
      5. Adjust f(1) to satisfy sum f(k)/k = 0.
      6. Verify M(x) <= 1.0001 for x in [1, 10K] using a vectorized difference array.
         M(x) = sum_{k=1}^K f(k) floor(x/k).
         We compute the increments of M(x) as x goes from 1 to 10K.
         M(x) - M(x-1) = sum_{k|x} f(k).
         This is equivalent to adding f(k) to all multiples of k.
         We use a difference array 'diff' where diff[j*k] += f(k).
         Then M(x) = cumsum(diff)[x].
         Complexity: O(K log K).
      7. If valid, return domain and values.
    """
    start_time = time.time()
    time_limit = 630.0
    safety_margin = 10.0

    K = 200000

    # 1. Compute Mobius function mu(k) for k=1..K
    mu = np.zeros(K + 1, dtype=np.int64)
    mu[1] = 1
    is_prime = np.ones(K + 1, dtype=bool)
    primes = []
    
    # Linear sieve
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

    domain = np.arange(1, K + 1, dtype=np.int64)
    values = mu[1:].astype(np.float64)

    # 2. Clip values to [-10, 10]
    values_clipped = np.clip(values, -10.0, 10.0)

    # 3. Adjust f(1)
    # S = sum f(k)/k
    S = np.sum(values_clipped / domain)
    values_adj = values_clipped.copy()
    values_adj[0] -= S  # f(1) is at index 0

    # 4. Verify M(x) <= 1.0001
    x_max = 10 * K
    # diff array for M(x)
    # M(x) = sum_{j=1}^x sum_{k|j} f(k)
    # Let g(j) = sum_{k|j} f(k). Then M(x) = sum_{j=1}^x g(j).
    # We can compute g(j) by iterating k and adding f(k) to g(j) for all j multiple of k.
    # This is the same as the difference array approach for M directly if we view M as a step function.
    # M(x) = sum_{k=1}^K f(k) floor(x/k).
    # floor(x/k) is 0 for x < k, 1 for k <= x < 2k, 2 for 2k <= x < 3k, etc.
    # So M(x) increases by f(k) at x = k, 2k, 3k, ...
    # Let diff be an array of size x_max + 2.
    # For each k, for j=1,2,... while j*k <= x_max:
    #   diff[j*k] += f(k)
    # Then M(x) = cumsum(diff)[x].
    
    diff = np.zeros(x_max + 2, dtype=np.float64)
    
    # Vectorized update:
    # For each k, indices are k, 2k, ..., floor(x_max/k)*k.
    # We can do this in a loop over k, which is K iterations.
    # Inside, we do a vector addition to diff[k::k] up to x_max.
    # This is efficient in NumPy.
    
    elapsed = time.time() - start_time
    if elapsed > time_limit - safety_margin:
        # Fallback
        domain_small = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values_small = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain_small, values_small

    for k in range(1, K + 1):
        f_k = values_adj[k - 1]
        if f_k == 0:
            continue
        # Add f_k to diff at indices k, 2k, ..., <= x_max
        # Slice: start=k, stop=x_max+1, step=k
        diff[k:x_max+1:k] += f_k
        if (k % 10000 == 0) and (time.time() - start_time > time_limit - safety_margin):
             domain_small = np.array([1, 2, 3, 5, 30], dtype=np.int64)
             values_small = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
             return domain_small, values_small

    M = np.cumsum(diff)
    
    # Check max M(x) for x in [1, x_max]
    # M[0] is 0. We care about x >= 1.
    max_M = np.max(M[1:x_max+1])
    
    if max_M > 1.0001:
        # Infeasible. This should not happen for K=200,000 with Mobius.
        # But if it does, we might try a smaller K or return fallback.
        # Let's try K = 100,000 just in case, though unlikely to be needed.
        pass # For now, fallback to be safe if constraint fails unexpectedly.
        domain_small = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values_small = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain_small, values_small

    # 5. Compute metric a
    log_k = np.log(domain)
    a = -np.sum(values_adj * log_k / domain)

    return domain, values

# EVOLVE-BLOCK-END
