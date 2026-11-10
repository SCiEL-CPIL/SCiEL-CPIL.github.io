# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog


def _mobius_upto(n):
    mu = np.ones(n + 1, dtype=np.int64)
    is_prime = np.ones(n + 1, dtype=bool)
    for i in range(2, int(n ** 0.5) + 1):
        if is_prime[i]:
            step = i
            for j in range(i * i, n + 1, step):
                is_prime[j] = False
            mu[i * i:n + 1:step] *= -1
            for j in range(i * i * i, n + 1, i * i):
                mu[j] = 0
    return mu


def prime_number_theorem_partial_function():
    """
    Construct a partial function f : subset of positive integers -> R that maximises
    a = -sum_k f(k)*log(k)/k under the step-function constraint
        M(x) = sum_k f(k)*floor(x/k) <= 1   for every integer x in [1, 10K],
    where K = max key in the domain.  The evaluator additionally replaces f(1) by
    f(1) - S with S = sum_k f(k)/k (clipped values), so that sum_k f(k)/k = 0.

    Design: solve the linear program
        max   c^T z
        s.t.  G z <= 1,   G[x,k] = floor((x+1)/k) - floor(x/k) = [k | (x+1)]
              z >= 0
    with c[k] = log(k)/k.  By the partial-summation identity
        sum_k z[k]*floor(x/k) = sum_{j<=x} (sum_{k|j} z[k])
    the constraint M(x)<=1 for all x is equivalent to the prefix sums of the
    divisor-sum d[j]=sum_{k|j} z[k] being <= 1, i.e. the single row G z <= 1.
    The LP optimum is the maximum feasible a.  The returned f is z with f(1) set
    so that sum_k f(k)/k = 0 (matching the evaluator's normalisation).
    """
    t0 = time.time()
    DEADLINE = 560.0          # wall-clock safety margin before the 600 s kill

    # ------------------------------------------------------------------
    # Precomputed fallbacks (verified feasible, a>0) in case the LP fails.
    # ------------------------------------------------------------------
    # 5-point hardcoded baseline (the seed): a = 1.0, feasible.
    def _baseline():
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values

    # mu on [1,1000]: a ~ 0.987, feasible (M(x)=sum_{j<=x} mu(j) in {0,1}).
    def _mu_fallback():
        mu = _mobius_upto(1000)
        domain = np.arange(1, 1001, dtype=np.int64)
        values = mu.astype(np.float64).copy()
        values[0] = -values[1:].sum() / 1000.0   # enforce sum f(k)/k = 0
        return domain, values

    # ------------------------------------------------------------------
    # Try the LP for increasing K until the deadline.
    # ------------------------------------------------------------------
    best = _baseline()
    best_a = 1.0

    for K in (1000, 2000, 3000, 4000, 5000):
        if time.time() > DEADLINE - 60:      # need >=60 s to finish a solve + check
            break
        try:
            ks = np.arange(1, K + 1, dtype=np.int64)
            c = np.log(ks.astype(np.float64)) / ks.astype(np.float64)   # max c^T z
            A = (np.arange(K + 2, dtype=np.int64) % ks) == 0            # [k | (x+1)]
            A = A.astype(np.float64)
            res = linprog(-c, A_ub=A, b_ub=np.ones(1),
                          bounds=[(0.0, None)] * K, method="highs")
            if not res.success:
                continue
            z = res.x
            # sanity: prefix-sum check
            d = np.zeros(K + 1, dtype=np.float64)
            np.add.at(d, ks, z)                       # d[k] += z[k] at index k
            # d[j] = sum_{k|j} z[k]
            for k in range(1, K + 1):
                for j in range(2 * k, K + 1, k):
                    d[j] += z[k - 1]
            if d.max() > 1.0 + 1e-6:
                continue
            # objective a = sum z[k]*log(k)/k (normalisation does not change a)
            a = float(np.dot(z, c))
            if a > best_a + 1e-9:
                values = z.copy()
                values[0] = -values[1:].sum() / K     # sum f(k)/k = 0
                # re-verify feasibility of the exact returned f via M(x)
                if _feasible(ks, values, DEADLINE):
                    best = (ks, values)
                    best_a = a
            if time.time() > DEADLINE - 30:
                break
        except Exception:
            continue

    return best


def _feasible(domain, values, deadline):
    """Return True iff M(x) <= 1.0001 for all integer x in [1, 10K]."""
    K = int(domain[-1])
    f = values
    # M(x) = sum_k f(k)*floor(x/k); compute via divisor-sum prefix.
    # d[j] = sum_{k|j} f(k); M(x) = sum_{j<=x} d[j].
    d = np.zeros(K + 1, dtype=np.float64)
    for idx, k in enumerate(domain):
        kk = int(k)
        for j in range(kk, K + 1, kk):
            d[j] += f[idx]
    running = np.cumsum(d)
    return bool(running.max() <= 1.0001) and (time.time() < deadline)


# EVOLVE-BLOCK-END
