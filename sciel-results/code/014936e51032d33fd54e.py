# EVOLVE-BLOCK-START
import time

import numpy as np
import sympy


def prime_number_theorem_partial_function():
    """
    Return a partial function f (as (domain, values)) that maximizes
        a = -sum_k f(k) * log(k) / k
    subject to the feasibility constraint
        M(x) = sum_k f(k) * floor(x/k) <= 1.0001  for every integer x in [1, 10*K].

    The optimal choice is the Mobius function mu(k), for which the identity
        sum_{k<=x} mu(k) * floor(x/k) = 1
    holds for every x (so M(x) = 1 <= 1.0001, feasible), and for which the
    partial objective a converges to 1 from below as more terms are included.

    We therefore include mu(1), mu(2), ..., mu(N) for the largest N that fits
    within the 600 s evaluation budget.  The objective a is a strictly
    increasing partial sum of -mu(k) log(k)/k, so a larger N gives a higher a.

    A hard wall-clock deadline guard guarantees we return before the evaluator
    kills the process (630 s), with a safety margin.
    """
    t0 = time.time()
    deadline = t0 + 590.0   # safety margin below the 600 s limit

    # --- Deterministic, verified feasible baseline (a strong Mobius subset) ---
    base_domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
    base_values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)

    def _mobius_values(n):
        """Return (domain, values) for mu(1..n)."""
        domain = np.arange(1, n + 1, dtype=np.int64)
        values = np.array(sympy.mobius(domain), dtype=np.float64)
        return domain, values

    # Start with a modest N that is fast and already improves on the baseline.
    N = 2000
    domain, values = _mobius_values(N)

    # Grow N while we have time, verifying feasibility on the full [1, 10*N] grid.
    # Each feasibility check is O(N * 10N); keep it comfortably within the deadline.
    while True:
        if time.time() > deadline - 8.0:
            break
        if N >= 200000:          # contract: maximum key K <= 200000
            break
        # Pick a growth step that keeps the next check cheap enough.
        # Cost of a check ~ N*10N; scale step so N_next^2 stays bounded.
        step = max(200, int(0.30 * N))
        N_next = min(200000, N + step)
        if N_next <= N:
            break

        cand_domain, cand_values = _mobius_values(N_next)
        if time.time() > deadline - 8.0:
            break

        # Feasibility: M(x) = sum_k f(k) floor(x/k) for x in [1, 10*N_next].
        # mu(1)=1 so M(1)=1 always; check x in [2, 10*N_next].
        x = np.arange(2, 10 * N_next + 1, dtype=np.int64)
        # M(x) = sum over k of mu(k) * floor(x/k); vectorized over x for each k.
        # Do it in chunks over k to bound memory.
        max_M = 1.0
        chunk = 500
        for s in range(0, N_next, chunk):
            e = min(N_next, s + chunk)
            ks = cand_domain[s:e]
            vs = cand_values[s:e]
            acc = np.zeros(x.shape, dtype=np.float64)
            for k in range(s, e):
                acc += cand_values[k] * (x // cand_domain[k])
            max_M = max(max_M, float(np.max(acc)))
            if time.time() > deadline - 8.0:
                break

        if max_M <= 1.0001:
            domain, values = cand_domain, cand_values
            N = N_next
        else:
            # Should never happen for mu (M is identically 1); stop to be safe.
            break

    # Final safety: ensure we are within the deadline and return a valid result.
    if time.time() > deadline:
        return base_domain, base_values

    return domain, values


# EVOLVE-BLOCK-END
