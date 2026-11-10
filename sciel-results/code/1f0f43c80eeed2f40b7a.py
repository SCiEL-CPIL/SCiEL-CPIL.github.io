# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_mobius(K):
    """Compute the Möbius function μ(n) for n = 1..K using a linear sieve."""
    mu = np.ones(K + 1, dtype=np.float64)
    is_prime = np.ones(K + 1, dtype=bool)
    primes = []
    mu[0] = 0.0
    is_prime[0] = False
    is_prime[1] = False

    for i in range(2, K + 1):
        if is_prime[i]:
            primes.append(i)
            mu[i] = -1.0
        for p in primes:
            prod = i * p
            if prod > K:
                break
            is_prime[prod] = False
            if i % p == 0:
                mu[prod] = 0.0
                break
            else:
                mu[prod] = -mu[i]
    return mu


def _check_feasibility(domain, values, K):
    """Check if M(x) <= 1.0001 for all x in [1, 10K]."""
    max_x = 10 * K
    g = np.zeros(max_x + 1, dtype=np.float64)
    for k, v in zip(domain, values):
        if v != 0.0:
            g[k::k] += v
    G = np.cumsum(g[1:])  # G[i] = M(i+1)
    return np.max(G) <= 1.0001


def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    """
    start_time = time.time()
    deadline = 600

    # The Möbius function μ(k) is the canonical choice.
    # a = -sum_k f(k) log(k)/k. For f(k)=μ(k), a = sum_{μ=-1} log(k)/k.
    # This is asymptotic to (6/pi^2) log K, which is the best known scaling.
    # Constraint M(x) <= 1.0001 is satisfied because M(x) = sum_{d|x} μ(d) = 1 if x=1 else 0,
    # so M(x) <= 1 for all x.
    # The normalization step: f(1) = f(1) - S where S = sum f(k)/k.
    # For f=μ, S = sum μ(k)/k ≈ 0 (it's 1/ζ(1) which diverges, but the partial sum converges to 0
    # as K→∞? No, sum_{k<=K} μ(k)/k = 1/ζ(1) is not right.
    # Actually sum_{k=1}^K μ(k)/k = 1/ζ(1)?? No.
    # Dirichlet series of μ is 1/ζ(s). At s=1 it has a pole.
    # The sum sum_{k=1}^K μ(k)/k is bounded. Specifically it is 1/ζ(1) + O(1/K) is wrong.
    # It is known that sum_{k<=x} μ(k)/k = O(1/x)? No.
    # sum_{k<=x} μ(k)/k = 1/ζ(1) is not defined.
    # The value is small. For K=200000, S is close to 0.
    # f_final(1) = μ(1) - S = 1 - S.
    # M(1) = f_final(1) = 1 - S.
    # We need 1 - S <= 1.0001 => S >= -0.0001.
    # Since S ≈ 0, this holds.
    
    K = 200000
    if time.time() - start_time > deadline:
        # Fallback to seed
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values

    mu = _compute_mobius(K)
    nonzero_mask = mu[1:] != 0
    domain = np.where(nonzero_mask)[0] + 1
    values = mu[1:][nonzero_mask]

    # Check feasibility
    if _check_feasibility(domain, values, K):
        return domain, values

    # Fallback to seed
    domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
    values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
    return domain, values


# EVOLVE-BLOCK-END
