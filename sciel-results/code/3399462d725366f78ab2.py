# EVOLVE-BLOCK-START
import numpy as np
import time


def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a = -sum_k f(k) * log(k) / k,
    subject to:
      - values clipped to [-10, 10]
      - f(1) adjusted so sum_k f(k)/k = 0
      - M(x) = sum_k f(k) * floor(x/k) <= 1.0001 for all x in [1, 10K]
    """
    start_time = time.time()
    time_limit = 630.0
    safety_margin = 15.0

    def elapsed():
        return time.time() - start_time

    def check_feasibility(domain, values):
        """Check if M(x) <= 1.0001 for all x in [1, 10*max(domain)]."""
        K = int(np.max(domain))
        x_max = 10 * K
        div_sum = np.zeros(x_max + 1, dtype=np.float64)
        for k, f_k in zip(domain, values):
            if f_k == 0.0:
                continue
            multiples = np.arange(k, x_max + 1, k)
            div_sum[multiples] += f_k
        M = np.cumsum(div_sum)
        return np.max(M[1:x_max + 1]) <= 1.0001

    def compute_metric(domain, values):
        """Compute a = -sum_k f(k) * log(k) / k."""
        log_k = np.log(domain.astype(np.float64))
        return -np.sum(values * log_k / domain.astype(np.float64))

    def finalize(domain, values):
        """Clip, adjust f(1), check feasibility, return (domain, values) or None."""
        values_clipped = np.clip(values, -10.0, 10.0)
        S = np.sum(values_clipped / domain.astype(np.float64))
        values_adj = values_clipped.copy()
        values_adj[0] = values_adj[0] - S
        if check_feasibility(domain, values_adj):
            return domain, values_adj
        return None

    def fallback():
        domain_small = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values_small = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain_small, values_small

    # Strategy: Use Mobius function for large K, but if infeasible, do a small-scale search.
    # First try K=200000 with mu.
    K = 200000
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
    result = finalize(domain, values)
    if result is not None:
        return result

    # Large K infeasible. Try smaller K values with mu.
    for K_try in [50000, 20000, 10000, 5000]:
        if elapsed() > time_limit - safety_margin:
            return fallback()
        domain_sub = np.arange(1, K_try + 1, dtype=np.int64)
        values_sub = mu[1:K_try + 1].astype(np.float64)
        result = finalize(domain_sub, values_sub)
        if result is not None:
            return result

    # If mu-based solutions are infeasible, do a small-scale search.
    # Search over K=2..15, values in {-1, 0, 1} for k >= 2.
    best_score = -np.inf
    best_result = None

    for K_small in range(2, 16):
        if elapsed() > time_limit - safety_margin:
            break
        domain_small = np.arange(1, K_small + 1, dtype=np.int64)
        n_vars = K_small - 1  # k=2..K_small
        # Enumerate all combinations of {-1, 0, 1} for k=2..K_small
        # Use itertools.product
        import itertools
        for combo in itertools.product([-1, 0, 1], repeat=n_vars):
            if elapsed() > time_limit - safety_margin:
                break
            values_small = np.zeros(K_small, dtype=np.float64)
            values_small[1:] = np.array(combo, dtype=np.float64)
            result = finalize(domain_small, values_small)
            if result is not None:
                score = compute_metric(domain_small, result[1])
                if score > best_score:
                    best_score = score
                    best_result = result

    if best_result is not None:
        return best_result

    return fallback()


# EVOLVE-BLOCK-END
