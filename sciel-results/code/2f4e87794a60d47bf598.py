import numpy as np
from math import log
import time

def mobius_up_to(n):
    """Compute Möbius function for 0..n using linear sieve."""
    mu = np.zeros(n + 1, dtype=np.int8)
    mu[1] = 1
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


def prime_number_theorem_partial_function():
    start_time = time.time()
    deadline = start_time + 620  # Safety margin before 630s limit

    max_K = 200000
    mu = mobius_up_to(max_K)

    # Seed's a value for normalization
    a_seed = -(0.0 + (-log(2) / 2) + (-log(3) / 3) + (-log(5) / 5) + log(30) / 30)

    best_score = -np.inf
    best_domain = None
    best_values = None

    # Try K from large to small; larger K gives larger a (closer to 1)
    K_candidates = [200000, 100000, 50000, 20000, 10000, 5000, 2000, 1000, 500, 200, 100, 50, 10]

    for K in K_candidates:
        if time.time() > deadline - 30:
            break

        # Domain: square-free integers in [1, K]
        mask = mu[1:K + 1] != 0
        domain = np.where(mask)[0] + 1
        values = mu[1:K + 1][mask].astype(np.float64)

        # Ensure 1 is in domain (mu(1) = 1, so it should be)
        if len(domain) == 0 or domain[0] != 1:
            domain = np.concatenate([[1], domain])
            values = np.concatenate([[1.0], values])

        # Simulate evaluator:
        # 1. Clip values to [-10, 10]
        values_clipped = np.clip(values, -10.0, 10.0)

        # 2. Compute S = sum f(k)/k over clipped values
        S = np.sum(values_clipped / domain)

        # 3. Adjust f(1): f(1) -= S
        idx1 = np.where(domain == 1)[0][0]
        values_clipped[idx1] -= S

        # 4. Compute M(x) = sum f(k) * floor(x/k) for x in [1, 10K]
        X_max = 10 * K
        diff = np.zeros(X_max + 2, dtype=np.float64)

        for k, fk in zip(domain, values_clipped):
            if fk == 0.0:
                continue
            k = int(k)
            multiples = np.arange(k, X_max + 1, k)
            diff[multiples] += fk
            ends = multiples + k
            valid = ends <= X_max + 1
            diff[ends[valid]] -= fk

        M = np.cumsum(diff[1:])
        max_M = np.max(M)

        # 5. Check feasibility
        if max_M > 1.0001:
            continue

        # 6. Compute objective a
        a = -np.sum(values_clipped * np.log(domain) / domain)
        score = a / a_seed

        if score > best_score:
            best_score = score
            best_domain = domain.copy()
            best_values = values.copy()

    # Fallback to seed if no feasible solution found
    if best_domain is None:
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values

    return best_domain.astype(np.int64), best_values.astype(np.float64)
