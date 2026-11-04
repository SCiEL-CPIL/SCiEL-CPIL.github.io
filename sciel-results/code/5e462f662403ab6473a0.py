"""Second autocorrelation inequality.
Structural design: sparse Sidon-like support + L-BFGS-B amplitude optimization.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 50
_DEADLINE = 110.0  # safety margin before the 120 s wall-clock kill


def _score(h):
    h = np.asarray(h, dtype=float)
    l1 = float(np.sum(h))
    if l1 <= 0.0:
        return 0.0
    l2sq = float(np.dot(h, h))
    sup = float(np.max(np.convolve(h, h)))
    if sup <= 0.0:
        return 0.0
    return l2sq / (l1 * sup)


def _sidon_support(m):
    """Greedy Sidon set (distinct pairwise sums) of size m in [0, N-1]."""
    pts = [0]
    used = set()
    for a in pts:
        for b in pts:
            used.add(a + b)
    for x in range(1, N):
        if len(pts) >= m:
            break
        ok = True
        for a in pts:
            if (a + x) in used:
                ok = False
                break
        if ok:
            pts.append(x)
            for a in pts:
                used.add(a + x)
    while len(pts) < m:
        # fallback: fill with largest unused indices
        used_idx = set(pts)
        for x in range(N - 1, -1, -1):
            if x not in used_idx:
                pts.append(x)
                break
    return sorted(pts[:m])


def _run_opt(support, log_a0, tol, max_iter):
    s = np.array(support, dtype=int)

    def fvec(x):
        a = np.exp(x)
        h = np.zeros(N)
        h[s] = a
        return h

    def objective(x):
        h = fvec(x)
        l1 = float(np.sum(h))
        if l1 <= 0.0:
            return 0.0
        l2sq = float(np.dot(h, h))
        sup = float(np.max(np.convolve(h, h)))
        if sup <= 0.0:
            return 0.0
        return -(l2sq / (l1 * sup))

    def jac(x):
        h = fvec(x)
        l1 = float(np.sum(h))
        if l1 <= 0.0:
            return np.zeros_like(x)
        l2sq = float(np.dot(h, h))
        conv = np.convolve(h, h)
        sup = float(np.max(conv))
        if sup <= 0.0:
            return np.zeros_like(x)
        val = l2sq / (l1 * sup)
        # dval/dh = 2h/(l1*sup) - (l2sq/l1^2)*(1/sup) - (l2sq/l1)*(1/sup^2)*d sup
        # d sup/dh_i = 2 * sum_j h_j * 1_{i+j in argmax}
        am = (conv == sup)
        d_sup = np.zeros(N)
        for i in range(N):
            d_sup[i] = 2.0 * float(np.sum(h[am] * 1.0)) if False else 0.0
        # efficient: d sup/dh_i = 2 * sum over j where (i+j) in argmax of h_j
        # conv index k = i + j, so for fixed i, j = k - i must be in [0,N-1]
        d_sup = np.zeros(N)
        for k in np.nonzero(am)[0]:
            jlo = max(0, k - (N - 1))
            jhi = min(N - 1, k)
            # i = k - j for j in [jlo, jhi]; d_sup[i] += 2*h[j]
            ii = k - np.arange(jlo, jhi + 1)
            jj = np.arange(jlo, jhi + 1)
            np.add.at(d_sup, ii, 2.0 * h[jj])
        dh = (2.0 * h) / (l1 * sup) - (l2sq / (l1 * l1)) * (1.0 / sup) \
            - (l2sq / l1) * (1.0 / (sup * sup)) * d_sup
        g = np.zeros_like(x)
        g[s] = dh[s] * a
        return g

    try:
        res = minimize(objective, log_a0, jac=jac, method='L-BFGS-B',
                       options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-6})
        if res.success or res.fun is not None:
            return res.x, -res.fun
    except Exception:
        pass
    # fallback: evaluate at start
    return log_a0, -objective(log_a0)


def construct():
    t0 = time.time()
    best_h = np.ones(N)
    best_score = _score(best_h)

    def remaining():
        return _DEADLINE - (time.time() - t0)

    # Candidate supports: various sparse Sidon sizes + a couple of dense-ish ones
    candidates = []
    for m in (4, 5, 6, 7, 8, 10, 12, 14, 16):
        candidates.append(_sidon_support(m))

    # Per-support initial log-amplitudes: try equal, then a few random patterns
    for support in candidates:
        if remaining() < 8.0:
            break
        m = len(support)
        inits = [np.zeros(m)]
        rng = np.random.default_rng(abs(hash(tuple(support))) % (2**32))
        for _ in range(3):
            inits.append(rng.uniform(-1.0, 1.0, size=m))
        for log_a0 in inits:
            if remaining() < 3.0:
                break
            log_a, sc = _run_opt(support, log_a0, tol=1e-10, max_iter=200)
            if sc > best_score:
                best_score = sc
                h = np.zeros(N)
                h[support] = np.exp(log_a)
                best_h = h.copy()

    # Final safety: ensure feasibility and return
    if np.any(best_h < 0):
        best_h = np.clip(best_h, 0.0, None)
    return [float(v) for v in best_h]
