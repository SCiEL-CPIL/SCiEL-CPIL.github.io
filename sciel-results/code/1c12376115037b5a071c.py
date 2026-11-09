"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize

_DEADLINE = time.time() + 325.0  # safety margin well inside the 330 s kill


def _finite_scores(mu, sigma, f_best, extra):
    """Assemble a finite, higher-is-better score array from a base and extra."""
    base = np.asarray(mu, dtype=float)
    sig = np.asarray(sigma, dtype=float)
    extra = np.asarray(extra, dtype=float)
    out = base + extra
    out = np.where(np.isfinite(out), out, 0.0)
    out = np.clip(out, -1e6, 1e6)
    if not np.all(np.isfinite(out)):
        out = np.zeros_like(out)
    return out


def _local_polish(score_arr, n_top, max_iters, time_budget):
    """Refine the top candidates with Nelder-Mead on the score surface.

    Returns (best_score, best_idx). Falls back to the top candidate if the
    refinement fails or the time budget is exhausted.
    """
    n = len(score_arr)
    if n < 3 or n_top < 1:
        top = int(np.argmax(score_arr))
        return float(score_arr[top]), top

    idx = np.argpartition(score_arr, -n_top)[-n_top:]
    base_score = float(score_arr[idx[0]])
    base_idx = int(idx[0])

    def neg_score(x):
        x = np.asarray(x, dtype=float)
        if not np.all(np.isfinite(x)) or np.any(x < 0.0) or np.any(x > 1.0):
            return 1e6
        x0 = x * (n - 1)
        i0 = int(np.clip(x0[0], 0, n - 1))
        i1 = int(np.clip(x0[1], 0, n - 1))
        w = x0[0] - i0
        if w < 0.0:
            i0 -= 1
            w += 1.0
        w = min(max(w, 0.0), 1.0)
        x1 = x * (n - 1)
        j0 = int(np.clip(x1[0], 0, n - 1))
        j1 = int(np.clip(x1[1], 0, n - 1))
        v = x1[1] - j0
        if v < 0.0:
            j0 -= 1
            v += 1.0
        v = min(max(v, 0.0), 1.0)
        val = (1.0 - w) * (1.0 - v) * score_arr[i0, j0] + \
              w * (1.0 - v) * score_arr[i1, j0] + \
              (1.0 - w) * v * score_arr[i0, j1] + \
              w * v * score_arr[i1, j1]
        return -val

    best_val = base_score
    best_idx = base_idx
    start = time.time()
    for i in idx:
        if time.time() > start + time_budget:
            break
        x0 = (np.array([i % n, i // n], dtype=float)) / max(n - 1, 1)
        try:
            res = minimize(neg_score, x0, method="Nelder-Mead",
                           options={"maxiter": max_iters, "xatol": 1e-4, "fatol": 1e-6})
            if res.success or np.isfinite(res.fun):
                val = -float(res.fun)
                if val > best_val + 1e-9:
                    best_val = val
                    best_idx = int(i)
        except Exception:
            continue
    return best_val, best_idx


def acquisition(mu, sigma, f_best):
    mu = np.asarray(mu, dtype=float).ravel()
    sigma = np.asarray(sigma, dtype=float).ravel()
    n = mu.size
    if n == 0:
        return np.zeros(0)
    if n < 2:
        return _finite_scores(mu, sigma, f_best, np.zeros(n))

    sig = np.clip(sigma, 1e-6, None)
    # Quantile expected improvement: slightly more robust than plain EI.
    z = (f_best - mu) / sig
    phi = norm.pdf(z)
    Phi = norm.cdf(z)
    q = 0.25
    qPhi = norm.cdf((z - q) / np.sqrt(1.0 + q * q))
    qz = (z - q) / np.sqrt(1.0 + q * q)
    ei = (f_best - mu) * qPhi + sig * norm.pdf(qz)
    ei = np.where(np.isfinite(ei), ei, 0.0)
    ei = np.clip(ei, -1e6, 1e6)

    # Entropy / uncertainty bonus to keep exploring where the GP is unsure.
    ent = np.where(sig > 0.0, sig * np.log(sig + 1e-12), 0.0)
    ent = np.where(np.isfinite(ent), ent, 0.0)

    # Blend: EI dominates, uncertainty nudges exploration in flat regions.
    base = ei + 0.05 * ent
    base = np.where(np.isfinite(base), base, 0.0)
    base = np.clip(base, -1e6, 1e6)

    # Bounded local polish on the top candidates (argmax is what matters).
    remaining = _DEADLINE - time.time()
    if remaining > 0.5:
        n_top = min(12, n)
        try:
            best_val, best_idx = _local_polish(base, n_top, max_iters=40,
                                               time_budget=min(remaining, 2.0))
            if best_val > base[best_idx] + 1e-9:
                extra = np.zeros(n)
                extra[best_idx] = best_val - base[best_idx]
                return _finite_scores(mu, sigma, f_best, extra)
        except Exception:
            pass

    return _finite_scores(mu, sigma, f_best, np.zeros(n))
# EVOLVE-BLOCK-END
