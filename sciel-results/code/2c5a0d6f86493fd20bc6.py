# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

_DEADLINE = 600.0
_MARGIN = 20.0
_START = time.time()

def _time_left():
    return _DEADLINE - _MARGIN - (time.time() - _START)

def _P_for_support(s, w):
    """Exact P[X1+X2+X3 < 2*X4] in float64."""
    s = np.asarray(s, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    n = s.size
    if n == 0 or n != w.size:
        return -np.inf
    if np.any(np.isnan(s)) or np.any(np.isinf(s)) or np.any(np.isnan(w)) or np.any(np.isinf(w)):
        return -np.inf
    if np.any(s < -1e-9) or np.any(w < -1e-9):
        return -np.inf
    tot = w.sum()
    if tot <= 1e-9:
        return -np.inf
    p = w / tot
    pair_sums = (s[:, None] + s[None, :]).ravel()
    pair_probs = (p[:, None] * p[None, :]).ravel()
    order = np.argsort(pair_sums, kind='mergesort')
    sorted_sums = pair_sums[order]
    cum = np.cumsum(pair_probs[order])
    thr = (2.0 * s[None, :] - s[:, None]).ravel()
    idx = np.searchsorted(sorted_sums, thr, side='left')
    F = np.zeros_like(idx)
    m = idx > 0
    F[m] = cum[idx[m] - 1]
    F = F.reshape(n, n)
    return float(np.sum((p[:, None] * p[None, :]) * F))

def _two_point():
    p = (15.0 - np.sqrt(33.0)) / 16.0
    s = np.array([0.0, 1.0], dtype=np.float64)
    w = np.array([p, 1.0 - p], dtype=np.float64)
    return s, w

def _three_point():
    """3-point: mass at 0, small value, large value. P includes (zero-count>=2) terms."""
    # s = [0, a, b], w = [p0, p1, p2]
    # P = sum over all 4-tuples where X1+X2+X3 < 2*X4
    # Let's just compute it numerically for a few candidates
    best = None
    best_P = -1.0
    for a in [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]:
        for b in [2.0, 3.0, 5.0, 8.0, 12.0, 20.0]:
            if b <= a:
                continue
            for p0 in [0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7]:
                p1 = (1.0 - p0) * 0.3
                p2 = 1.0 - p0 - p1
                if p2 < 0.01:
                    continue
                s = np.array([0.0, a, b], dtype=np.float64)
                w = np.array([p0, p1, p2], dtype=np.float64)
                P = _P_for_support(s, w)
                if P > best_P:
                    best_P = P
                    best = (s.copy(), w.copy())
    if best is not None:
        return best
    return _two_point()

def _pareto_model(params, n):
    """
    Pareto-quantile support with power-law weights.
    params: [log_alpha, log_xm, log_beta, log_p0]
    - alpha: Pareto shape (>1 for finite mean)
    - xm: scale parameter
    - beta: weight decay exponent (w_i ~ (i+1)^(-beta))
    - p0: weight of the first (zero) point
    """
    log_alpha, log_xm, log_beta, log_p0 = params
    alpha = float(np.exp(log_alpha))
    xm = float(np.exp(log_xm))
    beta = float(np.exp(log_beta))
    p0 = float(np.exp(log_p0))
    alpha = min(max(alpha, 1.05), 10.0)
    xm = min(max(xm, 0.01), 100.0)
    beta = min(max(beta, 0.5), 5.0)
    p0 = min(max(p0, 0.01), 0.95)

    # Support: s[0]=0, s[i] = xm * (1 - (1-u_i)^(1/alpha))^{-1} where u_i = i/n
    # Actually use Pareto quantile: Q(u) = xm * (1-u)^(-1/alpha)
    # Use u_i = i/(n) for i=1..n-1, avoid u=1
    u = np.linspace(0.01, 0.99, n - 1)
    s_tail = xm * (1.0 - u) ** (-1.0 / alpha)
    s = np.concatenate(([0.0], s_tail))
    s = np.maximum(s, 0.0)

    # Weights: w[0] = p0, w[i] proportional to (i+1)^(-beta) for i=1..n-1
    idx = np.arange(1, n, dtype=np.float64)
    w_tail = idx ** (-beta)
    w_tail = w_tail / w_tail.sum()
    w = np.concatenate(([p0], (1.0 - p0) * w_tail))
    w = w / w.sum()
    return s, w

def _objective(params, n):
    s, w = _pareto_model(params, n)
    P = _P_for_support(s, w)
    if P == -np.inf:
        return 1.0
    return -P

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    if _time_left() <= 0.0:
        return _two_point()

    best_P = -np.inf
    best_s = None
    best_w = None

    def _consider(s, w):
        nonlocal best_P, best_s, best_w
        s = np.asarray(s, dtype=np.float64)
        w = np.asarray(w, dtype=np.float64)
        if s.size == 0 or s.size != w.size:
            return
        if np.any(np.isnan(s)) or np.any(np.isinf(s)) or np.any(np.isnan(w)) or np.any(np.isinf(w)):
            return
        if np.any(s < -1e-9) or np.any(w < -1e-9):
            return
        if w.sum() <= 1e-9:
            return
        P = _P_for_support(s, w)
        if P != -np.inf and P > best_P:
            best_P, best_s, best_w = P, s.copy(), w.copy()

    # ---- 2-point baseline ----
    s2, w2 = _two_point()
    P2 = _P_for_support(s2, w2)
    if P2 > best_P:
        best_P, best_s, best_w = P2, s2.copy(), w2.copy()

    # ---- 3-point grid search ----
    if _time_left() > 5.0:
        s3, w3 = _three_point()
        _consider(s3, w3)

    # ---- Pareto model grid ----
    if _time_left() > 30.0:
        for alpha in [1.2, 1.5, 2.0, 3.0, 5.0]:
            for xm in [0.5, 1.0, 2.0, 5.0]:
                for beta in [1.0, 1.5, 2.0, 2.5, 3.0]:
                    for p0 in [0.3, 0.4, 0.5, 0.6, 0.7]:
                        if _time_left() <= 2.0:
                            break
                        n = 32
                        s, w = _pareto_model([np.log(alpha), np.log(xm), np.log(beta), np.log(p0)], n)
                        _consider(s, w)

    # ---- DE optimization over Pareto model ----
    bounds = [
        (np.log(1.05), np.log(10.0)),    # log_alpha
        (np.log(0.01), np.log(100.0)),   # log_xm
        (np.log(0.5), np.log(5.0)),      # log_beta
        (np.log(0.01), np.log(0.95)),    # log_p0
    ]
    for n in [16, 32, 64, 128]:
        if _time_left() <= 90.0:
            break
        try:
            res = differential_evolution(
                _objective,
                bounds,
                args=(n,),
                seed=42,
                maxiter=300,
                popsize=15,
                tol=1e-6,
                mutation=(0.5, 1.2),
                recombination=0.8,
                polish=True,
            )
            P_val = -res.fun
            if P_val > best_P:
                s, w = _pareto_model(res.x, n)
                _consider(s, w)
        except Exception:
            pass

    # ---- Local polish on best Pareto params if found ----
    if _time_left() > 40.0 and best_s is not None and len(best_s) > 2:
        try:
            n = len(best_s)
            # Infer Pareto params from best_s, best_w
            s = best_s
            w = best_w / best_w.sum()
            p0 = float(w[0])
            p0 = min(max(p0, 0.01), 0.95)
            # Estimate alpha from tail: s_i ~ xm * (1-u_i)^(-1/alpha)
            # Use last few points
            m = max(3, n // 4)
            tail_s = s[-m:]
            tail_u = np.linspace(0.99 - 1.0/m, 0.99, m)
            # log(s/xm) = -1/alpha * log(1-u)
            # Try a few xm values
            best_alpha_est = 2.0
            best_xm_est = 1.0
            best_err = 1e18
            for xm_try in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
                if xm_try >= tail_s[0] * 0.5:
                    continue
                log_ratio = np.log(tail_s / xm_try)
                log_inv_u = np.log(1.0 / (1.0 - tail_u))
                # log_ratio = -1/alpha * log_inv_u => 1/alpha = -log_ratio/log_inv_u
                valid = log_inv_u > 0.01
                if np.sum(valid) < 3:
                    continue
                est = -log_ratio[valid] / log_inv_u[valid]
                alpha_try = 1.0 / np.median(est)
                if alpha_try < 1.05 or alpha_try > 10.0:
                    continue
                # Compute residual
                pred = xm_try * (1.0 - tail_u) ** (-1.0 / alpha_try)
                err = np.sum((np.log(pred) - np.log(tail_s))**2)
                if err < best_err:
                    best_err = err
                    best_alpha_est = alpha_try
                    best_xm_est = xm_try
            # Estimate beta from weights
            w_tail = w[1:]
            w_tail_norm = w_tail / w_tail.sum()
            idx = np.arange(1, n, dtype=np.float64)
            log_w = np.log(np.maximum(w_tail_norm, 1e-300))
            log_i = np.log(idx)
            # w ~ i^(-beta) => log_w = -beta * log_i + const
            A = np.column_stack([log_i, np.ones_like(log_i)])
            try:
                coeffs, _, _, _ = np.linalg.lstsq(A, log_w, rcond=None)
                beta_est = -coeffs[0]
            except:
                beta_est = 2.0
            beta_est = min(max(beta_est, 0.5), 5.0)

            x0 = [np.log(best_alpha_est), np.log(best_xm_est), np.log(beta_est), np.log(p0)]
            res = minimize(_objective, x0, args=(n,), method='Nelder-Mead',
                           options={'maxiter': 5000, 'xatol': 1e-9, 'fatol': 1e-11})
            P_val = -res.fun
            if P_val > best_P:
                s, w = _pareto_model(res.x, n)
                _consider(s, w)
        except Exception:
            pass

    # ---- Final multi-start local polish ----
    if _time_left() > 20.0 and best_s is not None:
        try:
            n = len(best_s)
            # Try a few perturbations
            s_best = best_s.copy()
            w_best = best_w.copy()
            for trial in range(5):
                if _time_left() <= 10.0:
                    break
                # Perturb support
                s_try = s_best.copy()
                w_try = w_best.copy()
                if n > 2:
                    # Shift tail
                    shift = 1.0 + 0.2 * trial
                    s_try[1:] *= shift
                _consider(s_try, w_try)
        except Exception:
            pass

    if best_s is None:
        return _two_point()

    # Final validity
    if len(best_s) != len(best_w):
        return _two_point()
    if np.any(best_s < -1e-9) or np.any(best_w < -1e-9) or best_w.sum() <= 1e-9:
        return _two_point()
    if np.any(np.isnan(best_s)) or np.any(np.isinf(best_s)) or np.any(np.isnan(best_w)) or np.any(np.isinf(best_w)):
        return _two_point()
    return best_s, best_w

# EVOLVE-BLOCK-END
