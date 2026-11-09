# EVOLVE-BLOCK-START
import numpy as np
import time


def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.

    Strategy:
      1. Use the Mobius function mu(k) for k=1..K with K=200,000.
      2. Compute mu(k) via linear sieve.
      3. Clip values to [-10, 10] (mu values are already in {-1,0,1}, so no change).
      4. Adjust f(1) so that sum_k f(k)/k = 0.
      5. Verify M(x) = sum_k f(k) floor(x/k) <= 1.0001 for x in [1, 10K].
         Use difference array: M(x) - M(x-1) = sum_{k|x} f(k).
         Compute div_sum[n] = sum_{k|n} f(k) for n=1..10K by adding f(k) to multiples of k.
         Then M(x) = cumsum(div_sum)[x]. Complexity: O(K log K).
      6. If feasible, return (domain, values). Otherwise fall back to small solution.
    """
    start_time = time.time()
    time_limit = 630.0
    safety_margin = 15.0  # seconds

    def elapsed():
        return time.time() - start_time

    def fallback():
        domain_small = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values_small = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain_small, values_small

    K = 200000

    # 1. Compute Mobius function mu(k) for k=1..K using linear sieve
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

    if elapsed() > time_limit - safety_margin:
        return fallback()

    domain = np.arange(1, K + 1, dtype=np.int64)
    values = mu[1:].astype(np.float64)

    # 2. Clip values to [-10, 10]
    values_clipped = np.clip(values, -10.0, 10.0)

    # 3. Compute S = sum_k f(k)/k over clipped values
    S = np.sum(values_clipped / domain)

    # 4. Adjust f(1): f(1) = f(1)_clipped - S
    values_adj = values_clipped.copy()
    values_adj[0] = values_adj[0] - S

    if elapsed() > time_limit - safety_margin:
        return fallback()

    # 5. Verify M(x) <= 1.0001 for x in [1, 10K]
    # M(x) = sum_{k=1}^{K} f(k) * floor(x/k)
    # M(x) - M(x-1) = sum_{k|x} f(k)  (sum over k in domain that divide x)
    # Since domain is 1..K, div_sum[n] = sum_{k|n, k<=K} f(k)
    # For n > K, div_sum[n] = sum_{k|n, k<=K} f(k) (still only k up to K matter)
    # We need M(x) for x in [1, 10K], so compute div_sum for n=1..10K

    x_max = 10 * K
    div_sum = np.zeros(x_max + 1, dtype=np.float64)

    # For each k in domain, add f(k) to all multiples of k up to x_max
    # Use vectorized approach: for each k, indices = k, 2k, 3k, ..., floor(x_max/k)*k
    for k in range(1, K + 1):
        f_k = values_adj[k - 1]
        if f_k == 0.0:
            continue
        # Add f_k to div_sum at positions k, 2k, 3k, ..., up to x_max
        multiples = np.arange(k, x_max + 1, k)
        div_sum[multiples] += f_k

        if elapsed() > time_limit - safety_margin:
            return fallback()

    if elapsed() > time_limit - safety_margin:
        return fallback()

    # M(x) = cumsum(div_sum)[1:x_max+1] (M(0) = 0, M(x) for x>=1)
    M = np.cumsum(div_sum)
    # M[0] = 0, M[1] = div_sum[1], M[2] = div_sum[1]+div_sum[2], etc.
    # We need M(x) for x in [1, x_max], which is M[1:x_max+1]

    max_M = np.max(M[1:x_max + 1])

    if elapsed() > time_limit - safety_margin:
        return fallback()

    if max_M > 1.0001:
        # Infeasible. Try to find a smaller K that works.
        # Binary search on K to find the largest feasible K.
        # But first, let's try a few smaller values.
        for K_try in [100000, 50000, 20000, 10000, 5000, 1000]:
            if elapsed() > time_limit - safety_margin:
                return fallback()

            # Recompute mu up to K_try
            mu_sub = mu[:K_try + 1].copy()
            domain_sub = np.arange(1, K_try + 1, dtype=np.int64)
            values_sub = mu_sub[1:].astype(np.float64)
            values_sub_clipped = np.clip(values_sub, -10.0, 10.0)
            S_sub = np.sum(values_sub_clipped / domain_sub)
            values_sub_adj = values_sub_clipped.copy()
            values_sub_adj[0] = values_sub_adj[0] - S_sub

            x_max_sub = 10 * K_try
            div_sum_sub = np.zeros(x_max_sub + 1, dtype=np.float64)
            for k in range(1, K_try + 1):
                f_k = values_sub_adj[k - 1]
                if f_k == 0.0:
                    continue
                multiples = np.arange(k, x_max_sub + 1, k)
                div_sum_sub[multiples] += f_k

            M_sub = np.cumsum(div_sum_sub)
            max_M_sub = np.max(M_sub[1:x_max_sub + 1])

            if max_M_sub <= 1.0001:
                # Feasible! Return this solution.
                # Compute metric a = -sum f(k) log(k)/k
                log_k = np.log(domain_sub)
                a = -np.sum(values_sub_adj * log_k / domain_sub)
                return domain_sub, values_sub_adj

        # If no smaller K works, fall back
        return fallback()

    # Feasible with K=200000. Return the solution.
    return domain, values_adj


# EVOLVE-BLOCK-END
