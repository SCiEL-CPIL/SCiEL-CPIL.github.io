# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy import optimize
from scipy.optimize import minimize_scalar


def _compute_P(s, w):
    """Exact P[X1+X2+X3 < 2*X4] for support s, normalized weights w (strict <).
    
    P = sum_{k,l} w_k * w_l * G(2*s_l - s_k)
    where G(t) = sum_{i,j: s_i+s_j < t} w_i*w_j.
    
    O(n^2 log n^2) via precomputed sorted pairwise sums.
    """
    n = s.size
    if n == 1:
        return 0.0  # s+s+s < 2*s => s < 0, impossible since s>=0
    ps = s[:, None] + s[None, :]
    pw = w[:, None] * w[None, :]
    pf = ps.ravel()
    wf = pw.ravel()
    order = np.argsort(pf, kind='mergesort')
    pf_s = pf[order]
    cum = np.cumsum(wf[order])
    # Vectorize over all (k, l) pairs
    T = 2.0 * s[None, :] - s[:, None]  # T[k, l] = 2*s_l - s_k
    idx = np.searchsorted(pf_s, T.ravel(), side='left')
    # G_values[k, l] = cum[idx-1] if idx>0 else 0
    G_vals = np.zeros(T.size, dtype=np.float64)
    mask = idx > 0
    if mask.any():
        G_vals[mask] = cum[idx[mask] - 1]
    G_vals = G_vals.reshape(n, n)
    P = np.sum(w[:, None] * w[None, :] * G_vals)
    return float(P)


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


def _P_two_point(p, a, b):
    """Compute P for a 2-point distribution with support {a, b} and weight p on a."""
    s_arr = np.array([a, b])
    w_arr = np.array([p, 1.0 - p])
    return _compute_P(s_arr, w_arr)


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    t0 = time.time()
    deadline = 580.0  # safety margin (evaluator kills at 630s)
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
        if np.any(s < -1e-9) or np.any(w < -1e-9):
            return
        P = _compute_P(s, w)
        if P > best_P + 1e-15:
            best_P = P
            best_s = s.copy()
            best_w = w.copy()
            best_params = params.copy()
            best_n = n

    def elapsed():
        return time.time() - t0

    # --- Phase 1: Analytical two-point solution ---
    # For 2-point distribution with support {a, b} where a < b and weights {p, 1-p}
    for a in [0.0, 1e-6, 1e-4, 1e-3, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0]:
        for b in [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 500.0, 1000.0, 5000.0, 10000.0]:
            if b - a < 1e-10:
                continue
            try:
                res_ps = minimize_scalar(lambda pp: -_P_two_point(pp, a, b),
                                         bounds=(1e-10, 1.0 - 1e-10), method='bounded',
                                         options={'xatol': 1e-15})
                p_opt = res_ps.x
                raw_s = np.array([_softplus_inv(max(a, 1e-12)), _softplus_inv(b)])
                raw_w = np.array([np.log(p_opt), np.log(1.0 - p_opt)])
                params = np.concatenate([raw_s, raw_w])
                consider(params, 2)
            except Exception:
                pass

    # Refine best 2-point with Nelder-Mead
    if best_params is not None and elapsed() < deadline - 30:
        try:
            res = optimize.minimize(_negP, best_params, args=(2,), method='Nelder-Mead',
                                    options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-12,
                                             'adaptive': True})
            if res.x is not None and np.all(np.isfinite(res.x)):
                consider(res.x, 2)
        except Exception:
            pass

    # --- Phase 2: 3-point distributions ---
    if elapsed() < deadline - 60:
        inits = []
        
        # Geometric spacing
        s_init = np.logspace(np.log10(1e-4), np.log10(100.0), 3)
        w_init = np.ones(3) / 3
        raw_s = _softplus_inv(s_init)
        raw_w = np.log(w_init)
        inits.append(np.concatenate([raw_s, raw_w]))
        
        # Based on best 2-point, add a third point at various positions
        if best_n == 2:
            s2, w2 = _decode(best_params, 2)
            for frac in [0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9]:
                new_s = s2[0] + frac * (s2[1] - s2[0])
                if new_s < 1e-9:
                    new_s = 1e-6
                s3 = np.sort(np.array([s2[0], s2[1], new_s]))
                w3 = np.array([w2[0] * 0.7, w2[1] * 0.7, 0.05])
                w3 = w3 / w3.sum()
                raw_s = _softplus_inv(s3)
                raw_w = np.log(w3)
                inits.append(np.concatenate([raw_s, raw_w]))
        
        # Also try with the larger point extended
        if best_n == 2:
            s2, w2 = _decode(best_params, 2)
            for ext in [0.5, 1.0, 2.0, 5.0, 10.0]:
                new_s = s2[1] * ext
                s3 = np.sort(np.array([s2[0], s2[1], new_s]))
                w3 = np.array([w2[0], w2[1] * 0.5, 0.1])
                w3 = w3 / w3.sum()
                raw_s = _softplus_inv(s3)
                raw_w = np.log(w3)
                inits.append(np.concatenate([raw_s, raw_w]))
        
        for params in inits:
            if elapsed() > deadline - 60:
                break
            try:
                res = optimize.minimize(_negP, params, args=(3,), method='Powell',
                                        options={'maxiter': 1000, 'xtol': 1e-10, 'ftol': 1e-11})
                if res.x is not None and np.all(np.isfinite(res.x)):
                    consider(res.x, 3)
            except Exception:
                pass

    # --- Phase 3: 4-point distributions ---
    if elapsed() < deadline - 45:
        inits = []
        
        s_init = np.logspace(np.log10(1e-4), np.log10(100.0), 4)
        w_init = np.ones(4) / 4
        raw_s = _softplus_inv(s_init)
        raw_w = np.log(w_init)
        inits.append(np.concatenate([raw_s, raw_w]))
        
        if best_n >= 2:
            s_b, w_b = _decode(best_params, best_n)
            # Extend by adding points
            for new_s in [s_b[0] * 0.1, s_b[-1] * 2, s_b[-1] * 5]:
                if new_s < 1e-9:
                    new_s = 1e-6
                s4 = np.sort(np.concatenate([s_b, [new_s]]))
                w4 = np.concatenate([w_b * 0.8, [0.1]])
                w4 = w4 / w4.sum()
                raw_s = _softplus_inv(s4)
                raw_w = np.log(w4)
                inits.append(np.concatenate([raw_s, raw_w]))
        
        for params in inits:
            if elapsed() > deadline - 45:
                break
            try:
                res = optimize.minimize(_negP, params, args=(4,), method='Powell',
                                        options={'maxiter': 800, 'xtol': 1e-10, 'ftol': 1e-11})
                if res.x is not None and np.all(np.isfinite(res.x)):
                    consider(res.x, 4)
            except Exception:
                pass

    # --- Phase 4: 5-point distributions ---
    if elapsed() < deadline - 35:
        inits = []
        
        s_init = np.logspace(np.log10(1e-4), np.log10(200.0), 5)
        w_init = np.ones(5) / 5
        raw_s = _softplus_inv(s_init)
        raw_w = np.log(w_init)
        inits.append(np.concatenate([raw_s, raw_w]))
        
        if best_n >= 2:
            s_b, w_b = _decode(best_params, best_n)
            for new_s in [s_b[0] * 0.05, s_b[-1] * 3, s_b[-1] * 10]:
                if new_s < 1e-9:
                    new_s = 1e-6
                s5 = np.sort(np.concatenate([s_b, [new_s]]))
                w5 = np.concatenate([w_b * 0.7, [0.15]])
                w5 = w5 / w5.sum()
                raw_s = _softplus_inv(s5)
                raw_w = np.log(w5)
                inits.append(np.concatenate([raw_s, raw_w]))
        
        for params in inits:
            if elapsed() > deadline - 35:
                break
            try:
                res = optimize.minimize(_negP, params, args=(5,), method='Powell',
                                        options={'maxiter': 600, 'xtol': 1e-10, 'ftol': 1e-11})
                if res.x is not None and np.all(np.isfinite(res.x)):
                    consider(res.x, 5)
            except Exception:
                pass

    # --- Phase 5: Random restarts ---
    n_restarts = 0
    for attempt in range(60):
        if elapsed() > deadline - 25:
            break
        n = 2 + (attempt % 4)  # cycle through 2, 3, 4, 5
        # Smarter random initialization
        if n == 2:
            raw_s = np.array([rng.uniform(-5, 0), rng.uniform(0, 6)])
        elif n == 3:
            raw_s = np.sort(rng.uniform(-4, 5, size=n))
        elif n == 4:
            raw_s = np.sort(rng.uniform(-3, 5, size=n))
        else:
            raw_s = np.sort(rng.uniform(-2, 4, size=n))
        raw_w = rng.uniform(-2.0, 2.0, size=n)
        params = np.concatenate([raw_s, raw_w])
        
        try:
            res = optimize.minimize(_negP, params, args=(n,), method='Powell',
                                    options={'maxiter': 400, 'xtol': 1e-9, 'ftol': 1e-10})
            if res.x is not None and np.all(np.isfinite(res.x)):
                consider(res.x, n)
            n_restarts += 1
        except Exception:
            pass

    # --- Phase 6: Final high-precision polish ---
    if best_params is not None and elapsed() < deadline - 10:
        # L-BFGS-B for smooth convergence
        try:
            res_lb = optimize.minimize(_negP, best_params, args=(best_n,),
                                       method='L-BFGS-B',
                                       options={'maxiter': 1000, 'gtol': 1e-15, 'ftol': 1e-15, 'maxls': 50})
            if res_lb.x is not None and np.all(np.isfinite(res_lb.x)):
                consider(res_lb.x, best_n)
        except Exception:
            pass
    
    # Nelder-Mead polish with very tight tolerances
    if best_params is not None and elapsed() < deadline - 3:
        try:
            resf = optimize.minimize(_negP, best_params, args=(best_n,),
                                     method='Nelder-Mead',
                                     options={'maxiter': 50000, 'xatol': 1e-15, 'fatol': 1e-15,
                                              'adaptive': True})
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
