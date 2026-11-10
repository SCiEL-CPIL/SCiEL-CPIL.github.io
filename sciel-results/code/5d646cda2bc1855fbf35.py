# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_mobius(N):
    """Compute Moebius function mu(k) for k = 1..N using linear sieve."""
    mu = np.zeros(N + 1, dtype=np.int8)
    mu[1] = 1
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
    return mu


def _compute_max_M(mu, K, max_x):
    """
    Compute max of M(x) = sum_{k=1}^{K} mu(k) * floor(x/k) for x in [1, max_x].
    Uses the identity: M(x) = sum_{n=1}^{x} g(n) where g(n) = sum_{k|n, k<=K} mu(k).
    Returns the maximum value of M(x).
    """
    # Compute g(n) = sum_{k|n, k<=K} mu(k) for n = 1..max_x
    g = np.zeros(max_x + 1, dtype=np.float64)
    # For each k with mu(k) != 0, add mu(k) to g(n) for multiples of k
    for k in range(1, K + 1):
        v = mu[k]
        if v != 0:
            g[k::k] += v

    # M(x) = cumsum(g)
    M = np.cumsum(g)
    # Return max M(x) for x in [1, max_x]
    return np.max(M[1:max_x + 1]) if max_x >= 1 else 0.0


def _compute_objective_mu(mu, K):
    """Compute sum_k mu(k) * log(k) / k for k=1..K (only non-zero mu)."""
    # Only sum over k where mu(k) != 0
    mask = mu[1:K + 1] != 0
    ks = np.where(mask)[0] + 1
    if len(ks) == 0:
        return 0.0
    mu_vals = mu[1:K + 1][mask].astype(np.float64)
    log_k = np.log(ks.astype(np.float64))
    return np.sum(mu_vals * log_k / ks.astype(np.float64))


def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.

    Returns:
        (domain, values): a tuple of two numpy arrays of equal length describing f
    """
    start_time = time.time()
    deadline = 600.0

    best_domain = None
    best_values = None
    best_a = -np.inf

    # Precompute mobius for the maximum K we'll try
    K_max = 200000
    mu_full = _compute_mobius(K_max)

    # For each K, we use f(k) = c * mu(k) where c is chosen so that
    # max_x M(x) = c * max_x sum mu(k) floor(x/k) <= 1.0001
    # We want to maximize a = -c * sum_k mu(k) log(k)/k
    # So we want c as large as possible, i.e., c = 1.0001 / max_x M_mu(x)
    # where M_mu(x) = sum_k mu(k) floor(x/k)

    # Strategy: For each K, compute:
    #   max_M_mu = max_{x in [1, 10K]} sum_{k=1}^K mu(k) floor(x/k)
    #   S = sum_{k=1}^K mu(k) log(k) / k
    #   If max_M_mu > 0: c = 1.0001 / max_M_mu, a = -c * S
    #   If max_M_mu <= 0: we can make c arbitrarily large? No, M(x) must be <= 1.0001
    #   Actually if max_M_mu <= 0, then for any c > 0, M(x) = c * M_mu(x) <= 0 <= 1.0001
    #   So we can make c as large as we want? But values are clipped to [-10, 10]!
    #   So c * mu(k) must be in [-10, 10] for all k. Since mu(k) in {-1, 0, 1},
    #   c must be <= 10. So c = 10 if max_M_mu <= 0.

    # But wait, the clipping happens BEFORE the f(1) adjustment. Let me re-read.
    # "values are clipped to [-10, 10]. Then f(1) is replaced by f(1) - S"
    # So we need to account for clipping. If c > 10, values get clipped.
    # Let's just try c up to 10 and see.

    # Actually, let's be more careful. Let's try several c values for each K.

    # First, let's find for which K the constraint is binding.
    # For small K, max_M_mu might be small, allowing larger c.

    # Let's do a two-phase approach:
    # Phase 1: For a range of K, compute max_M_mu and S, find promising (K, c) pairs
    # Phase 2: Verify feasibility for the best candidates

    # Precompute for each K: max_M_mu and S
    # But this is expensive for large K. Let's be selective.

    K_candidates = list(range(10, 500)) + list(range(500, 2000, 50)) + \
                   list(range(2000, 10000, 200)) + list(range(10000, 50000, 1000)) + \
                   list(range(50000, 200001, 5000))
    K_candidates = sorted(set(K_candidates))

    # For each K, compute the key quantities
    results = []
    for K in K_candidates:
        if time.time() > deadline - 60:
            break

        max_x = 10 * K
        mu_K = mu_full[1:K + 1]

        # Compute max_M_mu = max_{x in [1, max_x]} sum_{k=1}^K mu(k) floor(x/k)
        max_M_mu = _compute_max_M(mu_full, K, max_x)

        # Compute S = sum_{k=1}^K mu(k) log(k) / k
        S = _compute_objective_mu(mu_full, K)

        # Determine optimal c
        if max_M_mu > 1e-10:
            c_opt = 1.0001 / max_M_mu
            # But c * mu(k) must be in [-10, 10] before clipping
            # mu(k) in {-1, 0, 1}, so |c * mu(k)| = c for non-zero mu
            # So c <= 10
            c_opt = min(c_opt, 10.0)
        else:
            # max_M_mu <= 0, so any c >= 0 works for the M constraint
            # But c <= 10 due to clipping
            c_opt = 10.0

        # Compute objective a = -c * S
        a = -c_opt * S

        results.append((K, c_opt, max_M_mu, S, a))

    # Now verify feasibility for the top candidates
    # Sort by a descending
    results.sort(key=lambda x: x[4], reverse=True)

    # Verify top 100 candidates
    for K, c, max_M_mu, S, a in results[:100]:
        if time.time() > deadline - 30:
            break

        max_x = 10 * K
        mu_K = mu_full[1:K + 1]
        non_zero = np.where(mu_K != 0)[0] + 1
        if len(non_zero) == 0:
            continue

        domain = non_zero.astype(np.int64)
        values = (c * mu_K[non_zero - 1]).astype(np.float64)

        # Clip to [-10, 10]
        values = np.clip(values, -10.0, 10.0)

        # Verify feasibility
        if not _verify_feasibility(domain, values, max_x):
            continue

        # Compute actual objective with clipped values
        a_actual = _compute_objective(domain, values)

        if a_actual > best_a:
            best_a = a_actual
            best_domain = domain.copy()
            best_values = values.copy()

    # Fallback
    if best_domain is None:
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
    else:
        domain = best_domain
        values = best_values

    return domain, values


# EVOLVE-BLOCK-END
