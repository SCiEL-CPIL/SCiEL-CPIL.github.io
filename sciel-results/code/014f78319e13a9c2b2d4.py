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

      The ideal candidate is the Mobius function mu(k), for which:
        sum_k mu(k) floor(x/k) = 1 for all x >= 1 (exact identity),
        and -sum_k mu(k) log(k)/k converges to 1 as K -> infinity.

      We compute mu(k) for k=1..K using a sieve. Since the sum of mu(k)/k over all k
      is 0 in the limit (and the partial sums are small), the adjustment S = sum f(k)/k
      is small, and f(1) = -S keeps the constraint M(x) <= 1 approximately.

      To be safe, we verify M(x) <= 1.0001 for all x in [1, 10K] before returning.
      If infeasible, we fall back to a smaller K or adjust.

      Compute budget: 630s wall-clock. Sieve for K=200000 is fast (<1s).
      The M(x) check costs about m * 10K where m ~ K (since domain is 1..K).
      Actually m = K (all integers from 1 to K). So cost is K * 10K = 10 K^2.
      For K=200000, that's 10 * (2e5)^2 = 4e11 operations — too slow!

      We need a smarter approach. Note that M(x) = sum_{k=1}^{K} f(k) floor(x/k).
      We can compute this for all x in [1, 10K] in O(K log K) using the fact that
      floor(x/k) is piecewise constant. Specifically, for each k, floor(x/k) changes
      at multiples of k. We can use a difference array: for each k, add f(k) to
      positions [j*k, (j+1)*k - 1] for j=1,2,...,floor(10K/k). This is O(K log K)
      total because sum_k floor(10K/k) = 10K * H(10K) ~ 10K * log(10K).

      Let's use this approach.
    """
    start_time = time.time()
    time_limit = 630.0
    safety_margin = 30.0  # seconds

    # Start with a reasonable K. We'll try to go as large as possible.
    # K = 200000 is the max. Let's try it.
    K = 200000

    # Compute Mobius function mu(k) for k = 1..K using a sieve
    mu = np.ones(K + 1, dtype=np.int64)
    is_prime = np.ones(K + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
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
    
    # mu[1] = 1 by definition
    mu[1] = 1

    # Domain is 1..K, values are mu(k)
    domain = np.arange(1, K + 1, dtype=np.int64)
    values = mu[1:].astype(np.float64)

    # Clip values to [-10, 10]
    values_clipped = np.clip(values, -10.0, 10.0)

    # Compute S = sum_k f(k)/k over clipped values
    S = np.sum(values_clipped / domain)

    # Adjust f(1): f(1) = f(1)_clipped - S
    # Since 1 is in the domain at index 0
    values_adj = values_clipped.copy()
    values_adj[0] = values_adj[0] - S

    # Re-clip after adjustment? The task says: "values are clipped to [-10, 10]. 
    # Then f(1) is replaced by f(1) - S ... so that afterwards sum_k f(k)/k = 0."
    # The clipping happens BEFORE the adjustment. So we should NOT re-clip after adjustment.
    # But wait: "values are clipped to [-10, 10]. Then f(1) is replaced by f(1) - S"
    # So the final values used for M(x) and a are: clipped values, with f(1) adjusted.

    # Verify M(x) = sum_k f(k) floor(x/k) <= 1.0001 for all x in [1, 10K]
    # Use the difference array approach for O(K log K) time
    x_max = 10 * K
    
    elapsed = time.time() - start_time
    if elapsed > time_limit - safety_margin:
        domain_small = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values_small = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain_small, values_small

    # Use difference array: diff[i] stores the change at position i
    # M[x] = cumsum(diff)[x]
    # For each k with f_k != 0:
    #   For j = 1, 2, ..., floor(x_max/k):
    #     diff[j*k] += f_k
    #     diff[(j+1)*k] -= f_k  (if (j+1)*k <= x_max)
    # Then M = cumsum(diff)
    diff = np.zeros(x_max + 2, dtype=np.float64)
    
    for k in range(1, K + 1):
        f_k = values_adj[k - 1]
        if f_k == 0:
            continue
        max_j = x_max // k
        if max_j == 0:
            continue
        # pos1 = k, 2k, ..., max_j*k (all unique)
        pos1 = np.arange(1, max_j + 1) * k
        # Add f_k at each pos1
        diff[pos1] += f_k
        # pos2 = 2k, 3k, ..., (max_j+1)*k
        # For pos2 <= x_max, subtract f_k
        # pos2 = pos1 + k
        pos2 = pos1 + k
        # All pos2 values are unique too
        # Subtract f_k where pos2 <= x_max
        # Since pos2 = 2k, 3k, ..., (max_j+1)*k
        # pos2 <= x_max iff (j+1)*k <= x_max iff j+1 <= x_max/k iff j < x_max/k
        # max_j = floor(x_max/k), so j+1 <= max_j+1
        # (max_j+1)*k may be > x_max
        # Let's just check:
        mask = pos2 <= x_max
        if np.any(mask):
            diff[pos2[mask]] -= f_k
        # For pos2 == x_max + 1, we can subtract there too (harmless)
        mask2 = pos2 == x_max + 1
        if np.any(mask2):
            diff[pos2[mask2]] -= f_k
        
        if (k % 50000 == 0) and (time.time() - start_time > time_limit - safety_margin):
            domain_small = np.array([1, 2, 3, 5, 30], dtype=np.int64)
            values_small = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
            return domain_small, values_small

    # Compute M from diff
    M = np.cumsum(diff)
    
    # Check if any M(x) > 1.0001 for x in [1, x_max]
    max_M = np.max(M[1:x_max + 1])
    if max_M > 1.0001:
        domain_small = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values_small = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain_small, values_small

    # Compute the metric a = -sum_k f(k) log(k) / k
    # Using the adjusted values
    log_k = np.log(domain)
    a = -np.sum(values_adj * log_k / domain)

    return domain, values


# EVOLVE-BLOCK-END
