# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog

_DEADLINE = 560.0
_START = time.time()


def _elapsed():
    return time.time() - _START


def _compute_mobius(n):
    """Möbius function mu(1..n) via sieve. mu(1)=1."""
    mu = np.ones(n + 1, dtype=np.int8)
    is_prime = np.ones(n + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    primes = []
    for i in range(2, n + 1):
        if is_prime[i]:
            primes.append(i)
            mu[i] = -1
        for p in primes:
            if p * i > n:
                break
            is_prime[p * i] = False
            if i % p == 0:
                mu[p * i] = 0
                break
            else:
                mu[p * i] = -mu[i]
    return mu


def _feasible_M(f, K):
    """Compute M(x) for x=1..10K. f is an array of length K+1 (1-indexed)."""
    f = f.astype(np.float64)
    f[0] = 0.0
    f = f[1:]
    m = len(f)
    x = np.arange(1, 10 * K + 1, dtype=np.int64)
    # M(x) = sum_{k} f[k] * floor(x/k)
    # = sum over blocks of x where floor(x/k) is constant
    M = np.zeros(len(x), dtype=np.float64)
    for k in range(1, m + 1):
        fk = f[k - 1]
        if fk == 0.0:
            continue
        M += fk * (x // k)
    return M


def _compute_S(f, K):
    """S = sum f(k)/k over k in 1..K (f 1-indexed)."""
    k = np.arange(1, K + 1, dtype=np.float64)
    return np.dot(f[1:], 1.0 / k)


def _objective(f, K):
    """a = -sum f(k) log(k)/k over k in 1..K (f 1-indexed)."""
    k = np.arange(1, K + 1, dtype=np.float64)
    return -np.dot(f[1:], np.log(k) / k)


def _build_mobius(K):
    """Return f array (1-indexed, length K+1) from Möbius, clipped to [-10,10]."""
    mu = _compute_mobius(K)
    f = mu.astype(np.float64)
    f = np.clip(f, -10.0, 10.0)
    return f


def _lp_refine(f, K, time_budget):
    """
    Refine f via LP to maximize -sum f(k) log(k)/k subject to:
      - M(x) <= 1 for x in a subset of [1, 10K]
      - -10 <= f(k) <= 10
    Uses scipy linprog (HiGHS). Returns refined f (1-indexed) or None.
    """
    n = K
    # Objective: maximize -sum f(k) log(k)/k  =>  minimize sum f(k) log(k)/k
    c = np.zeros(n, dtype=np.float64)
    k = np.arange(1, n + 1, dtype=np.float64)
    c = np.log(k) / k  # minimize c^T f

    # Bounds
    bounds = [(-10.0, 10.0)] * n

    # Constraints: M(x) <= 1 for a sample of x in [1, 10K]
    # M(x) = sum_k f(k) floor(x/k)
    # To keep the LP small, sample x values.
    max_x = 10 * K
    # Sample: all x up to some limit, plus random/structured samples
    # We need enough constraints to ensure feasibility. Use a structured set.
    # Key: M(x) is piecewise constant in x (changes at multiples). 
    # The maximum of M(x) over [1, 10K] occurs at x = 10K or at points where floor(x/k) jumps.
    # Actually M(x) is non-decreasing? No, f(k) can be negative.
    # Let's sample densely enough.
    n_constraints = 0
    A_rows = []
    b_vals = []

    # Include x = 10K (the largest, often critical)
    # Include x = 1, 2, ..., up to some small limit
    # Include x at positions where floor(x/k) changes for small k

    # Strategy: sample x = 1..min(10K, 2000) densely, plus 10K, plus a few in between
    sample_xs = []
    if max_x <= 5000:
        sample_xs = list(range(1, max_x + 1))
    else:
        # Dense near 1
        sample_xs.extend(range(1, 2001))
        # Some in the middle
        sample_xs.extend(range(2000, max_x, max(1, (max_x - 2000) // 500)))
        # Always include 10K
        if 10 * K not in sample_xs:
            sample_xs.append(10 * K)
        # Include points just below 10K
        sample_xs.append(max(1, 10 * K - 1))
        sample_xs.append(max(1, 10 * K // 2))

    # Deduplicate and sort
    sample_xs = sorted(set(sample_xs))
    if len(sample_xs) > 3000:
        # Subsample
        idx = np.linspace(0, len(sample_xs) - 1, 3000, dtype=int)
        sample_xs = [sample_xs[i] for i in idx]
        sample_xs = sorted(set(sample_xs))

    for x in sample_xs:
        row = np.zeros(n, dtype=np.float64)
        for kk in range(1, n + 1):
            row[kk - 1] = float(x // kk)
        A_rows.append(row)
        b_vals.append(1.0)
        n_constraints += 1

    if n_constraints == 0:
        return None

    A = np.array(A_rows, dtype=np.float64)
    b = np.array(b_vals, dtype=np.float64)

    # Time guard
    if _elapsed() > _DEADLINE - 30:
        return None

    try:
        res = linprog(
            c=c,
            A_ub=A,
            b_ub=b,
            bounds=bounds,
            method='highs',
            options={'time_limit': max(10, _DEADLINE - _elapsed() - 60), 'presolve': True}
        )
    except Exception:
        return None

    if res.status not in (0, 1, 2, 3):  # 0=optimal, 1=infeasible, 2=unbounded, 3=iteration limit
        # status 3 = iteration limit: may still have a feasible point
        if res.x is None:
            return None

    f_new = np.zeros(K + 1, dtype=np.float64)
    f_new[1:] = np.clip(res.x, -10.0, 10.0)
    return f_new


def _verify_and_adjust(f, K):
    """
    Clip f to [-10,10], adjust f(1) so sum f(k)/k = 0, verify M(x) <= 1.0001.
    Returns (domain, values, feasible, a) or (None, None, False, 0).
    """
    f = np.clip(f, -10.0, 10.0).astype(np.float64)
    # Adjust f(1)
    S = _compute_S(f, K)
    f[1] -= S  # so new sum f(k)/k = S - S = 0
    # Re-clip after adjustment (f(1) might go out of range)
    f[1] = np.clip(f[1], -10.0, 10.0)

    # Build domain: all k where f(k) != 0 (or just all 1..K for simplicity)
    # Use all k=1..K to be safe (the evaluator sums over the domain)
    domain = np.arange(1, K + 1, dtype=np.int64)
    values = f[1:].copy()

    # Verify feasibility: M(x) for x in [1, 10K]
    M = _feasible_M(f, K)
    if np.max(M) > 1.0001:
        return None, None, False, 0.0

    a = _objective(f, K)
    return domain, values, True, a


def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.

    Returns:
        (domain, values): a tuple of two numpy arrays of equal length describing f.
    """
    global _START, _DEADLINE
    _START = time.time()
    _DEADLINE = _START + 560.0

    K = 200000

    # Step 1: Build Möbius-based f
    if _elapsed() > _DEADLINE - 10:
        # Fallback: return the seed
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values

    f_mob = _build_mobius(K)

    # Verify Möbius baseline
    domain, values, feasible, a_mob = _verify_and_adjust(f_mob, K)
    if not feasible:
        # Möbius should be feasible; if not, fall back to seed
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values

    best_domain, best_values, best_a = domain, values, a_mob

    # Step 2: Try LP refinement with a smaller K for tractability
    # Use a smaller K for the LP to keep it fast, then extend
    K_lp = 5000
    if _elapsed() < _DEADLINE - 60:
        f_mob_small = _build_mobius(K_lp)
        f_lp = _lp_refine(f_mob_small, K_lp, _DEADLINE - _elapsed())
        if f_lp is not None:
            domain_lp, values_lp, feasible_lp, a_lp = _verify_and_adjust(f_lp, K_lp)
            if feasible_lp and a_lp > best_a:
                best_domain, best_values, best_a = domain_lp, values_lp, a_lp

    # Step 3: Try a larger K for LP if time permits
    K_lp2 = 10000
    if _elapsed() < _DEADLINE - 60:
        f_mob_small2 = _build_mobius(K_lp2)
        f_lp2 = _lp_refine(f_mob_small2, K_lp2, _DEADLINE - _elapsed())
        if f_lp2 is not None:
            domain_lp2, values_lp2, feasible_lp2, a_lp2 = _verify_and_adjust(f_lp2, K_lp2)
            if feasible_lp2 and a_lp2 > best_a:
                best_domain, best_values, best_a = domain_lp2, values_lp2, a_lp2

    # Final safety: if best is still the seed, return it
    if best_domain is None or best_values is None:
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values

    return best_domain, best_values


# EVOLVE-BLOCK-END
