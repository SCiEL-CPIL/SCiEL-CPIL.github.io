"""Second autocorrelation inequality.
Structural design: sparse Sidon-like support + L-BFGS-B amplitude optimization.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 50
_DEADLINE = 140.0  # safety margin before the 150 s wall-clock kill


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


def _soft_max_and_grad(conv, beta):
    """Compute soft-max of conv and its gradient w.r.t. conv.
    Returns (sm, d_sm) where sm = soft_max(conv, beta), d_sm = d(sm)/d(conv).
    """
    cmax = np.max(conv)
    shifted = beta * (conv - cmax)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    sm = cmax + np.log(sum_exp) / beta
    d_sm = exp_shifted / sum_exp / beta
    return sm, d_sm


def _run_opt(support, log_a0, tol, max_iter, use_soft_max=True, beta=50.0):
    s = np.array(support, dtype=int)
    m = len(s)

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
        conv = np.convolve(h, h)
        if use_soft_max:
            sup, _ = _soft_max_and_grad(conv, beta)
        else:
            sup = float(np.max(conv))
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
        if use_soft_max:
            sup, d_sup_conv = _soft_max_and_grad(conv, beta)
            # d_sup/dh_i = sum_k d_sup_conv[k] * 2*h[k-i] for valid k-i
            d_sup = np.zeros(N)
            for i in range(N):
                k_lo = max(0, i)
                k_hi = min(2 * N - 2, i + N - 1)
                if k_lo > k_hi:
                    continue
                jj = np.arange(k_lo, k_hi + 1) - i
                d_sup[i] = 2.0 * float(np.sum(d_sup_conv[k_lo:k_hi + 1] * h[jj]))
        else:
            sup = float(np.max(conv))
            if sup <= 0.0:
                return np.zeros_like(x)
            am = (conv == sup)
            d_sup = np.zeros(N)
            for k in np.nonzero(am)[0]:
                jlo = max(0, k - (N - 1))
                jhi = min(N - 1, k)
                ii = k - np.arange(jlo, jhi + 1)
                jj = np.arange(jlo, jhi + 1)
                np.add.at(d_sup, ii, 2.0 * h[jj])
        val = l2sq / (l1 * sup)
        # f = l2sq / (l1 * sup)
        # df/dh_i = (2*h_i * l1 * sup - l2sq * (sup + l1 * d_sup/dh_i)) / (l1*sup)^2
        dh = (2.0 * h * l1 * sup - l2sq * (sup + l1 * d_sup)) / (l1 * sup) ** 2
        g = np.zeros_like(x)
        g[s] = dh[s] * np.exp(x)
        return g

    try:
        res = minimize(objective, log_a0, jac=jac, method='L-BFGS-B',
                       options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-6})
        if res.fun is not None:
            return res.x, -res.fun
    except Exception:
        pass
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
        if remaining() < 10.0:
            break
        m = len(support)
        inits = [np.zeros(m)]
        rng = np.random.default_rng(abs(hash(tuple(support))) % (2**32))
        for _ in range(4):
            inits.append(rng.uniform(-1.5, 1.5, size=m))
        for log_a0 in inits:
            if remaining() < 4.0:
                break
            # First pass with soft max
            log_a, sc = _run_opt(support, log_a0, tol=1e-10, max_iter=300,
                                 use_soft_max=True, beta=30.0)
            if sc > best_score:
                best_score = sc
                h = np.zeros(N)
                h[support] = np.exp(log_a)
                best_h = h.copy()
            # Second pass with exact max (polish)
            if remaining() > 4.0:
                log_a2, sc2 = _run_opt(support, log_a, tol=1e-12, max_iter=200,
                                       use_soft_max=False, beta=50.0)
                if sc2 > best_score:
                    best_score = sc2
                    h = np.zeros(N)
                    h[support] = np.exp(log_a2)
                    best_h = h.copy()

    # Final safety: ensure feasibility and return
    if np.any(best_h < 0):
        best_h = np.clip(best_h, 0.0, None)
    return [float(v) for v in best_h]
