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
    
    result = 0.0
    for l in range(n):
        inner_sum = 0.0
        for k in range(n):
            t = 2.0 * s[l] - s[k]
            idx = np.searchsorted(pf_s, t, side='left')
            if idx > 0:
                inner_sum += w[k] * cum[idx - 1]
        result += w[l] * inner_sum
    
    return float(result)


def _P_two_point(a, b, p):
    """Exact P for 2-point distribution: support {a, b}, weights {p, 1-p}.
    
    X_i in {a, b}. Let r = b/a if a > 0, else handle separately.
    We need X1+X2+X3 < 2*X4.
    
    Cases (X1,X2,X3) sum < 2*X4:
    If X4 = a: need X1+X2+X3 < 2a. Since X_i >= a, sum >= 3a > 2a. Never true.
    If X4 = b: need X1+X2+X3 < 2b.
    
    Let k = number of X_i equal to b (out of X1,X2,X3). Sum = k*b + (3-k)*a.
    Need k*b + (3-k)*a < 2b, i.e., (k-2)*b < (k-3)*a, i.e., (2-k)*b > (3-k)*a.
    
    k=0: 2b > 3a -> b > 1.5a
    k=1: b > 2a
    k=2: 0 > a -> never (a >= 0)
    k=3: -b > 0 -> never
    
    So:
    - If b > 2a: all of k=0,1 work. P = (1-p)^3 * (1-p) + 3p(1-p)^2 * (1-p) = (1-p)^4 + 3p(1-p)^3 = (1-p)^3(1-p+3p) = (1-p)^3(1+2p)
    - If 1.5a < b <= 2a: only k=0 works. P = (1-p)^4
    - If b <= 1.5a: P = 0
    """
    if a < 0 or b < 0 or p < 0 or p > 1:
        return 0.0
    if a == 0:
        if b <= 0:
            return 0.0
        # X_i in {0, b}. Need X1+X2+X3 < 2*X4.
        # X4=0: need sum < 0, never.
        # X4=b: need sum < 2b. Sum = k*b where k = # of b's in X1,X2,X3.
        # k=0: 0 < 2b, yes. k=1: b < 2b, yes. k=2: 2b < 2b, no (strict). k=3: 3b < 2b, no.
        # P = (1-p)^3*(1-p) + 3*p*(1-p)^2*(1-p) = (1-p)^4 + 3p(1-p)^3 = (1-p)^3(1+2p)
        return (1.0 - p)**3 * (1.0 + 2.0 * p)
    if b <= a:
        return 0.0
    ratio = b / a
    if ratio > 2.0:
        return (1.0 - p)**3 * (1.0 + 2.0 * p)
    elif ratio > 1.5:
        return (1.0 - p)**4
    else:
        return 0.0


def _decode(params, n):
    raw_s = params[:n]
    raw_w = params[n:2 * n]
    s = np.logaddexp(0.0, raw_s)          # softplus -> (0, inf), smooth
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
    deadline = 590.0  # safety margin
    rng = np.random.default_rng(20240517)

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

    # --- Phase 1: Exact 2-point search using closed form ---
    # For 2-point {a, b}, p: P = (1-p)^3*(1+2p) if b/a > 2, else (1-p)^4 if b/a > 1.5, else 0
    # Maximize (1-p)^3*(1+2p): d/dp = -3(1-p)^2(1+2p) + 2(1-p)^3 = (1-p)^2[-3(1+2p)+2(1-p)] = (1-p)^2[-3-6p+2-2p] = (1-p)^2[-1-8p]
    # That's always negative for p in [0,1], so max at p=0: P=1. But p=0 means all mass on a, degenerate.
    # Wait, we need p>0 for a valid 2-point. As p->0+, P->1. But the support must have a < b.
    # Actually, the constraint is just that weights are non-negative and sum to 1. p can be very small.
    # But we also need a >= 0. Let's try a very small a, large b, small p.
    # With a=0: P = (1-p)^3*(1+2p), max at p=0 gives 1. But p=0 means single point.
    # For p > 0: P < 1. The best 2-point has P close to 1 but not exactly.
    
    # Let's do a grid search over ratio r = b/a and p
    # For r > 2: P = (1-p)^3*(1+2p). Max at p=0 is 1, but let's find interior max.
    # f(p) = (1-p)^3(1+2p). f'(p) = (1-p)^2(-1-8p) < 0 for p >= 0. So decreasing, max at p=0.
    # This suggests the optimal 2-point has p very small and b/a very large.
    # But we're constrained by the encoding: s = softplus(raw_s) > 0, w = softmax(raw_w) > 0.
    # So we can approach p=0 and b/a -> infinity.
    
    # Let's just do numerical optimization on the 2-point case, but with a good starting point.
    # Start with a very small a, large b, small p.
    a0, b0, p0 = 1e-8, 100.0, 1e-6
    raw_s = np.array([_softplus_inv(a0), _softplus_inv(b0)])
    raw_w = np.array([np.log(p0), np.log(1.0 - p0)])
    params = np.concatenate([raw_s, raw_w])
    consider(params, 2)
    
    # Optimize 2-point
    if best_params is not None:
        try:
            res = optimize.minimize(_negP, best_params, args=(2,), method='Nelder-Mead',
                                    options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-12})
            if res.x is not None and np.all(np.isfinite(res.x)):
                consider(res.x, 2)
        except Exception:
            pass

    # --- Phase 2: Try 3-point distributions ---
    # Initialize with a very small point, a medium point, and a large point
    s_init = np.array([1e-6, 1.0, 100.0])
    w_init = np.array([1e-4, 0.5, 0.5 - 1e-4])
    raw_s = _softplus_inv(s_init)
    raw_w = np.log(w_init)
    params = np.concatenate([raw_s, raw_w])
    
    try:
        res = optimize.minimize(_negP, params, args=(3,), method='Powell',
                                options={'maxiter': 1000, 'xtol': 1e-10, 'ftol': 1e-12})
        if res.x is not None and np.all(np.isfinite(res.x)):
            consider(res.x, 3)
    except Exception:
        pass
    
    if best_params is not None and best_n == 3:
        try:
            res2 = optimize.minimize(_negP, best_params, args=(3,), method='Nelder-Mead',
                                     options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12})
            if res2.x is not None and np.all(np.isfinite(res2.x)):
                consider(res2.x, 3)
        except Exception:
            pass

    # --- Phase 3: Try 4-point distributions ---
    s_init4 = np.array([1e-6, 0.01, 1.0, 100.0])
    w_init4 = np.array([1e-4, 0.1, 0.4, 0.5 - 1e-4])
    raw_s4 = _softplus_inv(s_init4)
    raw_w4 = np.log(w_init4)
    params4 = np.concatenate([raw_s4, raw_w4])
    
    if time.time() - t0 < deadline - 60:
        try:
            res4 = optimize.minimize(_negP, params4, args=(4,), method='Powell',
                                     options={'maxiter': 500, 'xtol': 1e-10, 'ftol': 1e-12})
            if res4.x is not None and np.all(np.isfinite(res4.x)):
                consider(res4.x, 4)
        except Exception:
            pass
        
        if best_params is not None and best_n == 4:
            try:
                res4b = optimize.minimize(_negP, best_params, args=(4,), method='Nelder-Mead',
                                          options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-12})
                if res4b.x is not None and np.all(np.isfinite(res4b.x)):
                    consider(res4b.x, 4)
            except Exception:
                pass

    # --- Phase 4: Random restarts if time permits ---
    for attempt in range(10):
        if time.time() - t0 > deadline - 30:
            break
        n = 3
        raw_s = np.concatenate([np.array([-16.0]), rng.uniform(-2.0, 4.0, size=n-1)])
        raw_w = rng.uniform(-4.0, 0.0, size=n)
        params = np.concatenate([raw_s, raw_w])
        
        try:
            res = optimize.minimize(_negP, params, args=(n,), method='Powell',
                                    options={'maxiter': 200, 'xtol': 1e-9, 'ftol': 1e-10})
            if res.x is not None and np.all(np.isfinite(res.x)):
                consider(res.x, n)
        except Exception:
            pass

    # --- Final polish ---
    if best_params is not None and time.time() - t0 < deadline - 10:
        try:
            resf = optimize.minimize(_negP, best_params, args=(best_n,),
                                     method='Nelder-Mead',
                                     options={'maxiter': 10000, 'xatol': 1e-13, 'fatol': 1e-13})
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
