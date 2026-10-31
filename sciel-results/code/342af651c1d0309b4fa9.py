"""Optimize nonnegative step heights to maximize (||h||_2^2) / (||h||_1 * ||h*h||_inf)."""
import time
import numpy as np
from scipy.optimize import minimize

N = 50
# Equal intervals of [-1/4, 1/4]
T = np.linspace(-0.25, 0.25, N)
DT = T[1] - T[0]
SEED_SCORE = 1.0
DEADLINE = 150.0


def _score(h):
    """Raw objective value for a nonnegative height vector h."""
    h = np.asarray(h, dtype=float)
    s1 = h.sum()
    if s1 <= 0:
        return 0.0
    s2 = float(np.dot(h, h))
    c = np.convolve(h, h, mode='full')
    m = float(c.max())
    if m <= 0:
        return 0.0
    return s2 / (s1 * m)


def _neg_log_ratio(params):
    """Negative log ratio for L-BFGS-B. params = [t_k..., s_k..., w_k...]."""
    K = len(params) // 3
    t = params[:K]
    s = np.exp(params[K:2 * K])          # widths > 0
    w = np.exp(params[2 * K:3 * K])      # weights > 0
    h = np.zeros(N)
    for k in range(K):
        h += w[k] * np.exp(-0.5 * ((T - t[k]) / s[k]) ** 2)
    s1 = h.sum()
    if s1 <= 0:
        return 1e12
    s2 = float(np.dot(h, h))
    c = np.convolve(h, h, mode='full')
    m = float(c.max())
    if m <= 0:
        return 1e12
    return -(np.log(s2) - np.log(s1) - np.log(m))


def construct():
    t0 = time.time()
    best_h = np.ones(N)
    best_val = _score(best_h)

    def timed_out():
        return (time.time() - t0) > (DEADLINE - 5.0)

    rng = np.random.default_rng(12345)

    # Random restarts over number of blobs K
    for K in [1, 2, 3, 4, 5, 6, 7, 8, 10, 12]:
        if timed_out():
            break
        for r in range(3):
            if timed_out():
                break
            t = rng.uniform(-0.25, 0.25, K)
            s = np.exp(rng.uniform(np.log(DT * 0.5), np.log(0.25), K))
            w = np.exp(rng.uniform(-3.0, 1.0, K))
            x0 = np.concatenate([t, np.log(s), np.log(w)])

            try:
                res = minimize(
                    _neg_log_ratio, x0, method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-9}
                )
            except Exception:
                continue

            h = np.zeros(N)
            for k in range(K):
                h += np.exp(res.x[2 * K + k]) * np.exp(
                    -0.5 * ((T - res.x[k]) / np.exp(res.x[K + k])) ** 2)
            v = _score(h)
            if v > best_val:
                best_val = v
                best_h = h

    # Local refinement: coordinate / small-step perturbation on the best vector
    if not timed_out():
        cur = best_h.copy()
        curv = best_val
        for _ in range(60):
            if timed_out():
                break
            # random local perturbation
            cand = cur * (1.0 + 0.2 * rng.standard_normal(N))
            cand = np.clip(cand, 0.0, None)
            if cand.sum() <= 0:
                continue
            v = _score(cand)
            if v > curv:
                cur, curv = cand, v

    return [float(x) for x in cur]
