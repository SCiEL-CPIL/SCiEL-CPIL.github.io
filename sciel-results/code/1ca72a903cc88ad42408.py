# EVOLVE-BLOCK-START
import time
import numpy as np


def _pair_sums(support):
    """All pairwise sums s_i + s_j as an (n, n) array (i, j)."""
    return support[:, None] + support[None, :]


def _compute_P(support, weights):
    """Exact P[X1+X2+X3 < 2*X4] in float64 with strict inequality.

    P = sum_{k,l} w_k w_l * (sum_{i,j: s_i+s_j < 2 s_l - s_k} w_i w_j)
    """
    n = len(support)
    if n < 1:
        return 0.0
    ps = _pair_sums(support)
    ps_sorted = np.sort(ps, axis=None)  # flat, sorted
    # For each (k, l): threshold t = 2*s_l - s_k, count pairs with sum < t.
    # We need the weighted sum of w_i w_j over those pairs, not just a count.
    # Use a matrix of indicators: for each (k,l), indicator over (i,j).
    # To keep it fast, precompute weights outer and accumulate via searchsorted
    # on a sorted-by-sum ordering that also carries the (i,j) weight.
    w = weights
    w_outer = w[:, None] * w[None, :]           # (n, n) weight per (i, j)
    # Flatten pair sums and pair weights in the same order
    ps_flat = ps.ravel()                         # (n^2,)
    wo_flat = w_outer.ravel()                    # (n^2,)
    order = np.argsort(ps_flat, kind="mergesort")
    ps_sorted = ps_flat[order]
    wo_sorted = wo_flat[order]
    cum = np.concatenate(([0.0], np.cumsum(wo_sorted)))

    # thresholds shape (n, n): k axis = X4? map: k is index of s_k (X3?), l is X4
    # P = sum_{k,l} w_k w_l * F_l(k), F_l(k) = sum_{i,j: s_i+s_j < 2 s_l - s_k} w_i w_j
    # Build thresholds T[k, l] = 2*s_l - s_k
    T = 2.0 * support[None, :] - support[:, None]   # (k, l)
    T_flat = T.ravel()
    # strict < : index = number of elements with ps < T  -> searchsorted('left')
    idx = np.searchsorted(ps_sorted, T_flat, side="left")  # (n^2,)
    F_flat = cum[idx]                                     # (n^2,)
    F = F_flat.reshape(n, n)                               # (k, l)
    # P = sum_{k,l} w_k w_l F[k, l]
    P = np.sum(w[:, None] * w[None, :] * F)
    return float(P)


def _fallback():
    n = 32
    support = np.arange(n, dtype=np.float64)
    weights = np.ones(n, dtype=np.float64) / n
    return support, weights


def _param_to_measure(params, n):
    a = float(params[0])
    b = float(params[1])
    v = params[2:]
    grid = np.linspace(0.0, 1.0, n)
    support = a + b * grid
    v = v - np.max(v)
    e = np.exp(v)
    weights = e / np.sum(e)
    return support, weights


def _objective(params, n):
    support, weights = _param_to_measure(params, n)
    # validity guard (support >= 0, weights > 0 guaranteed by construction except a)
    if np.any(support < -1e-9) or np.any(weights < -1e-9) or np.sum(weights) <= 1e-9:
        return 0.0
    return _compute_P(support, weights)


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    deadline = time.time() + 600.0
    # Start from a strong baseline: uniform on {0,1,...,n-1}
    base_n = 32
    base_support = np.arange(base_n, dtype=np.float64)
    base_weights = np.ones(base_n, dtype=np.float64) / base_n
    base_P = _compute_P(base_support, base_weights)

    best = (base_P, base_support.copy(), base_weights.copy())

    try:
        from scipy.optimize import minimize

        n = base_n
        # initial params: a=0, b=n-1, v=0 (uniform)
        v0 = np.zeros(n)
        x0 = np.concatenate(([0.0, float(n - 1)], v0))

        # Bounds: a >= 0 (support non-negative), b > 0 (spread), v free
        a_min = 0.0
        a_max = 1e6
        b_min = 1e-3
        b_max = 1e6
        bounds = [(a_min, a_max), (b_min, b_max)] + [(-50.0, 50.0)] * n

        def neg_obj(x):
            return -_objective(x, n)

        # Time-boxed optimization: stop when close to deadline.
        # L-BFGS-B with a callback that raises to break out near deadline.
        state = {"done": False}

        def cb(xk):
            if time.time() > deadline - 60.0:
                state["done"] = True
                raise KeyboardInterrupt

        res = minimize(
            neg_obj, x0, method="L-BFGS-B",
            bounds=bounds, options={
                "maxiter": 2000,
                "maxls": 20,
                "ftol": 1e-12,
                "gtol": 1e-8,
                "maxfun": 20000,
            },
            callback=cb,
        )
        if not state["done"]:
            xopt = res.x
            P_opt = _objective(xopt, n)
            s_opt, w_opt = _param_to_measure(xopt, n)
            if (np.all(s_opt >= -1e-9) and np.all(w_opt >= -1e-9)
                    and np.sum(w_opt) > 1e-9 and np.isfinite(P_opt)):
                if P_opt > best[0]:
                    best = (P_opt, s_opt, w_opt)
    except Exception:
        pass

    # Final validity clamp and return
    support, weights = best[1], best[2]
    support = np.asarray(support, dtype=np.float64).ravel()
    weights = np.asarray(weights, dtype=np.float64).ravel()
    # Sanitize: clip tiny negatives, ensure positive sum, finite.
    support = np.where(np.isfinite(support), support, 0.0)
    weights = np.where(np.isfinite(weights), weights, 0.0)
    support = np.clip(support, 0.0, None)
    weights = np.clip(weights, 0.0, None)
    if weights.size != support.size or weights.size == 0:
        return _fallback()
    if np.sum(weights) <= 1e-9:
        return _fallback()
    return support, weights


# EVOLVE-BLOCK-END
