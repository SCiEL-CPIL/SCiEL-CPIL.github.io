# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy import optimize


def _compute_P(s, w):
    """Exact P[X1+X2+X3 < 2*X4] for support s (>=0), normalized weights w (strict <).

    P = sum_{k,l} w_k * w_l * G(2*s_l - s_k),  G(t) = sum_{i,j: s_i+s_j < t} w_i*w_j.
    O(n^2 log n^2) via precomputed sorted pairwise sums.
    """
    n = s.size
    if n == 1:
        return 0.0
    ps = s[:, None] + s[None, :]
    pw = w[:, None] * w[None, :]
    pf = ps.ravel()
    wf = pw.ravel()
    order = np.argsort(pf, kind='mergesort')
    pf_s = pf[order]
    cum = np.cumsum(wf[order])
    res = 0.0
    for l in range(n):
        inner = 0.0
        for k in range(n):
            t = 2.0 * s[l] - s[k]
            idx = np.searchsorted(pf_s, t, side='left')
            if idx > 0:
                inner += w[k] * cum[idx - 1]
        res += w[l] * inner
    return float(res)


def _decode(params, n):
    raw_s = params[:n]
    raw_w = params[n:2 * n]
    s = np.logaddexp(0.0, raw_s)          # softplus -> (0, inf)
    mx = np.max(raw_w)
    w = np.exp(raw_w - mx)
    w = w / w.sum()                        # softmax -> positive, sums to 1
    return s, w


def _negP(params, n):
    s, w = _decode(params, n)
    return -_compute_P(s, w)


def _softplus_inv(s):
    s = np.maximum(s, 1e-9)
    return np.log(np.expm1(s))


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    t0 = time.time()
    deadline = 580.0  # safety margin before the 630s kill

    best_P = -1.0
    best_s = None
    best_w = None
    best_params = None
    best_n = None

    def consider(params, n):
        nonlocal best_P, best_s, best_w, best_params, best_n
        s, w = _decode(params, n)
        if not (np.all(np.isfinite(s)) and np.all(np.isfinite(w))):
            return
        P = _compute_P(s, w)
        if P > best_P + 1e-13:
            best_P = P
            best_s = s.copy()
            best_w = w.copy()
            best_params = params.copy()
            best_n = n

    # --- Exact analytic two-point optimum: support {0, d}, weight p=0.75 at 0 ---
    # P = p^3 + 3*p^2*(1-p) = 0.84375  (maximized at p = 0.75)
    p2 = 0.75
    d2 = 1.0
    raw_s2 = np.array([_softplus_inv(1e-9), _softplus_inv(d2)])
    raw_w2 = np.array([np.log(p2), np.log(1.0 - p2)])
    consider(np.concatenate([raw_s2, raw_w2]), 2)

    # Refine the two-point solution (d should be free; P is scale-invariant in d)
    if best_params is not None and time.time() - t0 < deadline - 60:
        try:
            res = optimize.minimize(_negP, best_params, args=(2,), method='Nelder-Mead',
                                    options={'maxiter': 4000, 'xatol': 1e-12, 'fatol': 1e-12})
            if res.x is not None and np.all(np.isfinite(res.x)):
                consider(res.x, 2)
        except Exception:
            pass

    # --- Multi-point search: geometrically spaced supports, near-0 heavy weight ---
    for n_try in [3, 4, 5, 6]:
        if time.time() - t0 > deadline - 60:
            break
        # Base init: heavy weight near 0, decaying geometric support
        s_init = np.array([1e-8, 1e-4, 1e-2, 1.0, 10.0, 100.0][:n_try])
        w_init = np.array([0.75, 0.15, 0.06, 0.03, 0.008, 0.002][:n_try])
        w_init = w_init / w_init.sum()
        raw_s = _softplus_inv(s_init)
        raw_w = np.log(np.maximum(w_init, 1e-12))
        params = np.concatenate([raw_s, raw_w])
        consider(params, n_try)

        if time.time() - t0 > deadline - 45:
            break
        try:
            res = optimize.minimize(_negP, params, args=(n_try,), method='Powell',
                                    options={'maxiter': 400, 'xtol': 1e-9, 'ftol': 1e-11})
            if res.x is not None and np.all(np.isfinite(res.x)):
                consider(res.x, n_try)
        except Exception:
            pass

        if best_n == n_try and time.time() - t0 < deadline - 30:
            try:
                res2 = optimize.minimize(_negP, best_params, args=(n_try,), method='Nelder-Mead',
                                         options={'maxiter': 2000, 'xatol': 1e-11, 'fatol': 1e-11})
                if res2.x is not None and np.all(np.isfinite(res2.x)):
                    consider(res2, n_try)
            except Exception:
                pass

    # --- Random restarts for small n ---
    rng = np.random.default_rng(20240517)
    for attempt in range(30):
        if time.time() - t0 > deadline - 30:
            break
        n = 3 if attempt % 3 == 0 else (4 if attempt % 3 == 1 else 5)
        # Geometric-ish random supports, log-spaced
        s_rand = np.exp(rng.uniform(np.log(1e-8), np.log(50.0), size=n))
        s_rand = np.sort(s_rand)
        w_rand = rng.dirichlet(np.ones(n) * 0.5)
        # bias weight toward the smallest support
        w_rand = w_rand * np.exp(-0.3 * np.arange(n))
        w_rand = w_rand / w_rand.sum()
        raw_s = _softplus_inv(s_rand)
        raw_w = np.log(np.maximum(w_rand, 1e-12))
        params = np.concatenate([raw_s, raw_w])
        try:
            res = optimize.minimize(_negP, params, args=(n,), method='Powell',
                                    options={'maxiter': 250, 'xtol': 1e-8, 'ftol': 1e-10})
            if res.x is not None and np.all(np.isfinite(res.x)):
                consider(res.x, n)
        except Exception:
            pass

    # Final high-precision polish
    if best_params is not None and time.time() - t0 < deadline - 8:
        try:
            resf = optimize.minimize(_negP, best_params, args=(best_n,),
                                     method='Nelder-Mead',
                                     options={'maxiter': 12000, 'xatol': 1e-13, 'fatol': 1e-13})
            if resf.x is not None and np.all(np.isfinite(resf.x)):
                consider(resf.x, best_n)
        except Exception:
            pass

    s = np.ascontiguousarray(best_s, dtype=np.float64).ravel()
    w = np.ascontiguousarray(best_w, dtype=np.float64).ravel()
    s = np.maximum(s, 0.0)
    w = np.maximum(w, 0.0)
    w = w / w.sum()
    assert s.size == w.size >= 1
    assert np.all(np.isfinite(s)) and np.all(np.isfinite(w))
    assert np.all(s >= -1e-9) and np.all(w >= -1e-9) and w.sum() > 1e-9
    return s, w

# EVOLVE-BLOCK-END
