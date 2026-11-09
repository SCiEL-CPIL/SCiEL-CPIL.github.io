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
    # = sum_k w_k * G_k where G_k = sum_l w_l * G(2*s_l - s_k)
    # Alternatively: P = sum_l w_l * sum_k w_k * G(2*s_l - s_k)
    # Let's compute for each l: H_l = sum_k w_k * G(2*s_l - s_k)
    # Then P = sum_l w_l * H_l
    
    # For efficiency, compute G at all needed thresholds
    # Thresholds are 2*s_l - s_k for all k,l -> n^2 values
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


def _negP(params, n):
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
    """Optimize the 2-point case: P = p_a^3 + 3 p_a^2 p_b + 3 p_a p_b^2 * [indicator conditions].
    
    With support {a, b} and weights {p, 1-p} (a < b), we need to figure out which
    triples satisfy X1+X2+X3 < 2*X4.
    
    X4 can be a or b.
    If X4 = a: need X1+X2+X3 < 2a. Since a >= 0, this requires all of X1,X2,X3 = a (since smallest is a, sum >= 3a > 2a unless a=0).
    Actually if a > 0: 3a < 2a means a < 0, impossible. So if a > 0, no triple works for X4=a.
    If a = 0: need X1+X2+X3 < 0, impossible since all >= 0.
    So X4=a never works (for a >= 0).
    
    If X4 = b: need X1+X2+X3 < 2b.
    Possible triples:
    - (a,a,a): sum = 3a < 2b iff a < 2b/3
    - (a,a,b): sum = 2a+b < 2b iff 2a < b iff a < b/2
    - (a,b,b): sum = a+2b < 2b iff a < 0, impossible for a >= 0
    - (b,b,b): sum = 3b < 2b iff b < 0, impossible
    
    So P = p_b * [p_a^3 * I(a < 2b/3) + 3 p_a^2 p_b * I(a < b/2)]
       = (1-p) * p^3 * I(a < 2b/3) + 3(1-p) p^2 (1-p) * I(a < b/2)
       where p = weight of a, (1-p) = weight of b.
    
    To maximize, we want both indicators true: a < b/2 (which implies a < 2b/3).
    Then P = (1-p) p^3 + 3(1-p)^2 p^2 = p^2(1-p)[p + 3(1-p)] = p^2(1-p)(3-2p)
    
    Maximize f(p) = p^2(1-p)(3-2p) = p^2(3 - 5p + 2p^2) = 3p^2 - 5p^3 + 2p^4
    f'(p) = 6p - 15p^2 + 8p^3 = p(6 - 15p + 8p^2)
    8p^2 - 15p + 6 = 0 -> p = (15 ± sqrt(225-192))/16 = (15 ± sqrt(33))/16
    p = (15 - sqrt(33))/16 ≈ (15-5.7446)/16 ≈ 0.5785
    
    f(p) at this point gives the max P for 2-point.
    """
    p = (15.0 - np.sqrt(33.0)) / 16.0
    # Need a < b/2. Choose a and b freely as long as a >= 0, b > 2a.
    # The value of P doesn't depend on actual a, b (only on weights) when both indicators are 1.
    # So we can pick a=0, b=1 for simplicity.
    a, b = 0.0, 1.0
    return np.array([a, b]), np.array([p, 1.0 - p])


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

    # Build a diverse set of initializations
    inits = []
    # 2-point analytic
    inits.append((s0.copy(), w0.copy()))

    # 3-point: try to add a middle point
    for a in [0.0, 0.01, 0.05]:
        for b in [0.5, 1.0, 2.0]:
            for c in [1.0, 3.0, 5.0]:
                if a < b < c:
                    for pw in [(0.5, 0.3, 0.2), (0.4, 0.4, 0.2), (0.5, 0.2, 0.3)]:
                        inits.append((np.array([a, b, c]), np.array(pw)))

    # 4-point
    for arr in [np.array([0.0, 0.25, 0.5, 1.0]),
                np.array([0.0, 0.1, 0.5, 2.0]),
                np.array([0.0, 0.5, 1.5, 4.0])]:
        for pw in [(0.4, 0.3, 0.2, 0.1), (0.5, 0.25, 0.15, 0.1)]:
            inits.append((arr.copy(), np.array(pw)))

    # 5-point
    for arr in [np.linspace(0, 2, 5), np.array([0.0, 0.1, 0.5, 1.5, 4.0])]:
        for pw in [np.array([0.4, 0.25, 0.15, 0.1, 0.1]), np.array([0.5, 0.2, 0.1, 0.1, 0.1])]:
            inits.append((arr.copy(), pw.copy()))
    
    # 6-point
    for arr in [np.linspace(0, 3, 6), np.array([0.0, 0.1, 0.3, 0.8, 2.0, 5.0])]:
        for pw in [np.array([0.35, 0.2, 0.15, 0.1, 0.1, 0.1]),
                   np.array([0.4, 0.2, 0.15, 0.1, 0.1, 0.05])]:
            inits.append((arr.copy(), pw.copy()))

    # Evaluate all initializations
    for s_init, w_init in inits:
        consider(s_init, w_init)

    # Optimize each initialization using Nelder-Mead on the parameter space
    # We'll optimize in the (raw_s, raw_w) space with softplus/softmax decoding
    best_params = None
    best_n_params = None

    def _negP_params(params, n):
        s, w = _decode(params, n)
        if not (np.all(np.isfinite(s)) and np.all(np.isfinite(w))):
            return 1e10
        if np.any(s < -1e-9) or np.any(w < -1e-9):
            return 1e10
        P = _compute_P(s, w)
        return -P

    # Build param vectors from initializations
    param_inits = []
    for s_init, w_init in inits:
        n = s_init.size
        raw_s = _softplus_inv(s_init)
        raw_w = np.log(w_init + 1e-15)
        params = np.concatenate([raw_s, raw_w])
        param_inits.append((params, n))

    # Add some random initializations
    for n in [2, 3, 4, 5, 6]:
        for _ in range(4):
            s_rand = np.sort(rng.uniform(0.0, 3.0, size=n))
            w_rand = rng.dirichlet(np.ones(n) * 2.0)
            raw_s = _softplus_inv(s_rand)
            raw_w = np.log(w_rand + 1e-15)
            param_inits.append((np.concatenate([raw_s, raw_w]), n))

    # Optimize each
    for idx, (params, n) in enumerate(param_inits):
        if time.time() - t0 > deadline - 30:
            break
        try:
            res = optimize.minimize(_negP_params, params, args=(n,), method='Nelder-Mead',
                                    options={'maxiter': 5000, 'xatol': 1e-11, 'fatol': 1e-13,
                                             'adaptive': True})
            if res.x is not None and np.all(np.isfinite(res.x)):
                s_opt, w_opt = _decode(res.x, n)
                consider(s_opt, w_opt)
        except Exception:
            pass

    # Local refinement of the best found so far with perturbation-based escape
    if best_s is not None and time.time() - t0 < deadline - 20:
        n_best = best_s.size
        raw_s = _softplus_inv(best_s)
        raw_w = np.log(best_w + 1e-15)
        params_best = np.concatenate([raw_s, raw_w])
        
        # Try multiple perturbation-based refinements
        for pert_scale in [0.01, 0.005, 0.001]:
            if time.time() - t0 > deadline - 10:
                break
            perturbed = params_best + rng.normal(0, pert_scale, size=params_best.size)
            try:
                res = optimize.minimize(_negP_params, perturbed, args=(n_best,), method='Nelder-Mead',
                                        options={'maxiter': 8000, 'xatol': 1e-12, 'fatol': 1e-14,
                                                 'adaptive': True})
                if res.x is not None and np.all(np.isfinite(res.x)):
                    s_fin, w_fin = _decode(res.x, n_best)
                    consider(s_fin, w_fin)
            except Exception:
                pass
        
        # Final high-precision polish on the best found
        if time.time() - t0 < deadline - 10:
            # Re-encode from current best
            raw_s = _softplus_inv(best_s)
            raw_w = np.log(best_w + 1e-15)
            params_best = np.concatenate([raw_s, raw_w])
            try:
                res = optimize.minimize(_negP_params, params_best, args=(n_best,), method='Nelder-Mead',
                                        options={'maxiter': 15000, 'xatol': 1e-14, 'fatol': 1e-16,
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
