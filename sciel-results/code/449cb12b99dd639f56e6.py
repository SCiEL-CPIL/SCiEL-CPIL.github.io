"""Optimized program: search for a height vector maximizing the autoconvolution ratio."""
import time
import numpy as np

N = 50
_DEADLINE = 110.0  # safety margin under 120s
_t0 = None

def _ratio(h):
    h = np.asarray(h, dtype=float)
    s1 = h.sum()
    if s1 <= 0:
        return 0.0
    s2 = float(np.dot(h, h))
    # autoconvolution over all lags (nonnegative h => max at lag 0, but compute properly)
    # full linear autoconvolution
    ac = np.correlate(h, h, mode='full')
    peak = float(ac.max())
    if peak <= 0:
        return 0.0
    return s2 / (s1 * peak)

def construct():
    global _t0
    if _t0 is None:
        _t0 = time.time()

    best = np.ones(N)
    best_r = _ratio(best)

    # Candidate: concentrated spike with tiny tail
    spike = np.zeros(N)
    spike[0] = 1.0
    best2 = spike.copy()
    r2 = _ratio(best2)
    if r2 > best_r:
        best, best_r = best2, r2

    # Try to improve via scipy if available and time permits
    try:
        from scipy.optimize import minimize
        def neglog(h):
            r = _ratio(h)
            if r <= 0:
                return 1e6
            return -np.log(r)
        # several starts
        starts = [
            np.ones(N),
            np.zeros(N),
            np.linspace(1.0, 0.1, N),
            np.exp(-np.linspace(0, 4, N)),
        ]
        starts[1][0] = 1.0
        for x0 in starts:
            if time.time() - _t0 > _DEADLINE - 5:
                break
            try:
                res = minimize(neglog, x0, method='L-BFGS-B',
                               bounds=[(0.0, None)] * N,
                               options={'maxiter': 400, 'maxfun': 4000})
                if res.success or res.fun < neglog(best):
                    cand = np.clip(res.x, 0.0, None)
                    rc = _ratio(cand)
                    if rc > best_r:
                        best, best_r = cand, rc
            except Exception:
                pass
    except Exception:
        pass

    # Ensure nonnegative and finite
    best = np.clip(np.asarray(best, dtype=float), 0.0, None)
    if not np.all(np.isfinite(best)) or best.sum() <= 0:
        best = np.zeros(N); best[0] = 1.0

    return [float(x) for x in best]
