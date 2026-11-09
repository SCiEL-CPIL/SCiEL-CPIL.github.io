# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy import optimize


def _compute_P(s, w):
    """Exact P[X1+X2+X3 < 2*X4] for support s, normalized weights w (strict <)."""
    n = s.size
    if n <= 1:
        return 0.0
    # Build all n^2 pair sums (for X1+X2)
    ps = s[:, None] + s[None, :]
    pw = w[:, None] * w[None, :]
    psf = ps.ravel()
    pwf = pw.ravel()
    order = np.argsort(psf, kind='stable')
    psf_s = psf[order]
    cum = np.cumsum(pwf[order])
    # G(T) = weight of pairs with sum strictly < T
    # For each k,l: need pair sum < 2*s_l - s_k
    # P = sum_k w_k * sum_l w_l * G(2*s_l - s_k)
    thresholds = 2.0 * s[None, :] - s[:, None]  # shape (n_k, n_l)
    idx = np.searchsorted(psf_s, thresholds.ravel(), side='left')
    G_vals = np.zeros(n * n)
    nz = idx > 0
    G_vals[nz] = cum[idx[nz] - 1]
    G_matrix = G_vals.reshape(n, n)  # G_matrix[k, l] = G(2*s_l - s_k)
    # H_l = sum_k w_k * G_matrix[k, l]
    H = w @ G_matrix  # shape (n,) -> H[l]
    return float(np.sum(w * H))


def _decode(params, n):
    raw_s = params[:n]
    raw_w = params[n:2 * n]
    s = np.logaddexp(0.0, raw_s)
    mx = np.max(raw_w)
    w = np.exp(raw_w - mx)
    w = w / w.sum()
    return s, w


def _negP_params(params, n):
    s, w = _decode(params, n)
    if not (np.all(np.isfinite(s)) and np.all(np.isfinite(w))):
        return 1e10
    if np.any(s < -1e-9) or np.any(w < -1e-9):
        return 1e10
    P = _compute_P(s, w)
    return -P


def _softplus_inv(s):
    s = np.maximum(s, 1e-9)
    return np.log(np.expm1(s))


def _two_point_optimize():
    """Optimize the 2-point case analytically."""
    p = (15.0 - np.sqrt(33.0)) / 16.0
    a, b = 0.0, 1.0
    return np.array([a, b]), np.array([p, 1.0 - p])


def _three_point_analytic():
    """Try to find good 3-point configurations analytically.
    
    With support {a, b, c} (a < b < c) and weights {p1, p2, p3}:
    X4 = a: need X1+X2+X3 < 2a. Only possible if all = a (3a < 2a -> a < 0, impossible) or a=0 and all < 0 (impossible). So 0.
    X4 = b: need X1+X2+X3 < 2b.
      - (a,a,a): 3a < 2b iff a < 2b/3
      - (a,a,b): 2a+b < 2b iff a < b
      - (a,b,b): a+2b < 2b iff a < 0, impossible
      - others: even larger
    X4 = c: need X1+X2+X3 < 2c.
      - (a,a,a): 3a < 2c (usually true)
      - (a,a,b): 2a+b < 2c (usually true)
      - (a,a,c): 2a+c < 2c iff a < c
      - (a,b,b): a+2b < 2c iff a < 2c-2b
      - (a,b,c): a+b+c < 2c iff a+b < c
      - (a,c,c): a+2c < 2c iff a < 0, impossible
      - (b,b,b): 3b < 2c iff b < 2c/3
      - (b,b,c): 2b+c < 2c iff b < c
      - (b,c,c): b+2c < 2c iff b < 0, impossible
      - (c,c,c): 3c < 2c, impossible
    
    This is complex; let's just use numerical optimization with good inits.
    """
    results = []
    # Try various 3-point configurations
    configs = [
        (np.array([0.0, 0.5, 1.0]), np.array([0.4, 0.35, 0.25])),
        (np.array([0.0, 0.3, 1.0]), np.array([0.45, 0.3, 0.25])),
        (np.array([0.0, 0.4, 1.5]), np.array([0.4, 0.35, 0.25])),
        (np.array([0.0, 0.25, 0.8]), np.array([0.5, 0.25, 0.25])),
        (np.array([0.0, 0.5, 2.0]), np.array([0.4, 0.3, 0.3])),
        (np.array([0.0, 0.33, 1.0]), np.array([0.45, 0.3, 0.25])),
    ]
    for s, w in configs:
        P = _compute_P(s, w)
        results.append((P, s.copy(), w.copy()))
    results.sort(key=lambda x: -x[0])
    return results[0][1], results[0][2]


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    t0 = time.time()
    deadline = 590.0
    rng = np.random.default_rng(20240517)

    best_P = -1.0
    best_s = None
    best_w = None

    def consider(s, w):
        nonlocal best_P, best_s, best_w
        s = np.asarray(s, dtype=np.float64).ravel()
        w = np.asarray(w, dtype=np.float64).ravel()
        n = s.size
        if n < 1 or s.size != w.size:
            return
        if not (np.all(np.isfinite(s)) and np.all(np.isfinite(w))):
            return
        if np.any(s < -1e-9) or np.any(w < -1e-9):
            return
        if w.sum() <= 1e-9:
            return
        w_norm = w / w.sum()
        P = _compute_P(s, w_norm)
        if P > best_P + 1e-14:
            best_P = P
            best_s = s.copy()
            best_w = w_norm.copy()

    # Start with the analytic 2-point optimum
    s0, w0 = _two_point_optimize()
    consider(s0, w0)

    # Try 3-point analytic
    s3, w3 = _three_point_analytic()
    consider(s3, w3)

    # Build a diverse set of initializations
    inits = []
    # 2-point analytic
    inits.append((s0.copy(), w0.copy()))
    # 3-point best
    inits.append((s3.copy(), w3.copy()))

    # More 3-point variations
    for a in [0.0, 0.01, 0.05, 0.1]:
        for b in [0.2, 0.3, 0.4, 0.5, 0.6]:
            for c in [0.8, 1.0, 1.2, 1.5, 2.0]:
                if a < b < c:
                    for pw in [(0.4, 0.35, 0.25), (0.45, 0.3, 0.25), (0.5, 0.25, 0.25), (0.35, 0.35, 0.3)]:
                        inits.append((np.array([a, b, c]), np.array(pw)))

    # 4-point
    for arr in [np.array([0.0, 0.2, 0.5, 1.0]),
                np.array([0.0, 0.15, 0.4, 1.0]),
                np.array([0.0, 0.25, 0.6, 1.5]),
                np.array([0.0, 0.1, 0.3, 0.8]),
                np.array([0.0, 0.3, 0.7, 1.5])]:
        for pw in [(0.35, 0.3, 0.2, 0.15), (0.4, 0.25, 0.2, 0.15), (0.3, 0.3, 0.25, 0.15)]:
            inits.append((arr.copy(), np.array(pw)))

    # 5-point
    for arr in [np.array([0.0, 0.15, 0.35, 0.6, 1.0]),
                np.array([0.0, 0.1, 0.3, 0.6, 1.2]),
                np.array([0.0, 0.2, 0.4, 0.7, 1.3])]:
        for pw in [np.array([0.3, 0.25, 0.2, 0.15, 0.1]),
                   np.array([0.35, 0.25, 0.2, 0.1, 0.1]),
                   np.array([0.3, 0.3, 0.2, 0.1, 0.1])]:
            inits.append((arr.copy(), pw.copy()))

    # 6-point
    for arr in [np.array([0.0, 0.1, 0.25, 0.45, 0.7, 1.1]),
                np.array([0.0, 0.15, 0.3, 0.5, 0.8, 1.4]),
                np.array([0.0, 0.2, 0.35, 0.55, 0.8, 1.3])]:
        for pw in [np.array([0.25, 0.2, 0.18, 0.15, 0.12, 0.1]),
                   np.array([0.3, 0.2, 0.18, 0.14, 0.1, 0.08])]:
            inits.append((arr.copy(), pw.copy()))

    # 7-point
    for arr in [np.array([0.0, 0.1, 0.2, 0.35, 0.5, 0.7, 1.0]),
                np.array([0.0, 0.12, 0.25, 0.4, 0.55, 0.75, 1.1])]:
        for pw in [np.array([0.22, 0.18, 0.16, 0.14, 0.12, 0.1, 0.08]),
                   np.array([0.25, 0.18, 0.15, 0.13, 0.11, 0.09, 0.09])]:
            inits.append((arr.copy(), pw.copy()))

    # Evaluate all initializations
    for s_init, w_init in inits:
        consider(s_init, w_init)

    # Build param vectors from initializations
    param_inits = []
    for s_init, w_init in inits:
        n = s_init.size
        raw_s = _softplus_inv(s_init)
        raw_w = np.log(w_init + 1e-15)
        params = np.concatenate([raw_s, raw_w])
        param_inits.append((params, n))

    # Add some random initializations
    for n in [3, 4, 5, 6, 7]:
        for _ in range(3):
            s_rand = np.sort(rng.uniform(0.0, 2.5, size=n))
            w_rand = rng.dirichlet(np.ones(n) * 2.0)
            raw_s = _softplus_inv(s_rand)
            raw_w = np.log(w_rand + 1e-15)
            param_inits.append((np.concatenate([raw_s, raw_w]), n))

    # Sort param inits by initial P (best first) to prioritize good starting points
    def _init_score(item):
        params, n = item
        s, w = _decode(params, n)
        if np.any(s < -1e-9) or np.any(w < -1e-9) or not np.all(np.isfinite(s)) or not np.all(np.isfinite(w)):
            return -1e10
        return _compute_P(s, w)

    param_inits.sort(key=lambda x: -_init_score(x))

    # Optimize each
    for idx, (params, n) in enumerate(param_inits):
        if time.time() - t0 > deadline - 30:
            break
        try:
            res = optimize.minimize(_negP_params, params, args=(n,), method='Nelder-Mead',
                                    options={'maxiter': 6000, 'xatol': 1e-11, 'fatol': 1e-13,
                                             'adaptive': True})
            if res.x is not None and np.all(np.isfinite(res.x)):
                s_opt, w_opt = _decode(res.x, n)
                consider(s_opt, w_opt)
        except Exception:
            pass

    # Local refinement of the best found so far
    if best_s is not None and time.time() - t0 < deadline - 20:
        n_best = best_s.size
        raw_s = _softplus_inv(best_s)
        raw_w = np.log(best_w + 1e-15)
        params_best = np.concatenate([raw_s, raw_w])
        
        # Try multiple perturbation-based refinements
        for pert_scale in [0.01, 0.005, 0.002, 0.001]:
            if time.time() - t0 > deadline - 10:
                break
            perturbed = params_best + rng.normal(0, pert_scale, size=params_best.size)
            try:
                res = optimize.minimize(_negP_params, perturbed, args=(n_best,), method='Nelder-Mead',
                                        options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-14,
                                                 'adaptive': True})
                if res.x is not None and np.all(np.isfinite(res.x)):
                    s_fin, w_fin = _decode(res.x, n_best)
                    consider(s_fin, w_fin)
            except Exception:
                pass
        
        # Final high-precision polish on the best found
        if time.time() - t0 < deadline - 10:
            raw_s = _softplus_inv(best_s)
            raw_w = np.log(best_w + 1e-15)
            params_best = np.concatenate([raw_s, raw_w])
            try:
                res = optimize.minimize(_negP_params, params_best, args=(n_best,), method='Nelder-Mead',
                                        options={'maxiter': 20000, 'xatol': 1e-14, 'fatol': 1e-16,
                                                 'adaptive': True})
                if res.x is not None and np.all(np.isfinite(res.x)):
                    s_fin, w_fin = _decode(res.x, n_best)
                    consider(s_fin, w_fin)
            except Exception:
                pass

    # Final validation
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
