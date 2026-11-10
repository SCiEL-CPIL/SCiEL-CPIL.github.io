# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

_HARD_DEADLINE = time.time() + 615


def _compute_P_vectorized(support, weights):
    n = len(support)
    w = weights / weights.sum()
    if n <= 1:
        return 0.0
    S = support[:, None] + support[None, :]
    W = np.outer(w, w)
    thresholds = 2.0 * support[None, :] - support[:, None]
    S_4d = S[None, None, :, :]
    thr_4d = thresholds[:, :, None, None]
    mask = S_4d < thr_4d
    W_4d = W[None, None, :, :]
    inner = np.einsum('klmn,klmn->kl', mask, W_4d)
    P = np.sum(w[:, None] * w[None, :] * inner)
    return float(P)


def _compute_P_fast(support, weights):
    n = len(support)
    w = weights / weights.sum()
    if n <= 1:
        return 0.0
    S = support[:, None] + support[None, :]
    W = np.outer(w, w)
    S_flat = S.ravel()
    W_flat = W.ravel()
    order = np.argsort(S_flat, kind='mergesort')
    S_sorted = S_flat[order]
    W_sorted = W_flat[order]
    W_cdf = np.cumsum(W_sorted)
    P = 0.0
    for k in range(n):
        for l in range(n):
            thr = 2.0 * support[l] - support[k]
            idx = np.searchsorted(S_sorted, thr, side='left')
            if idx > 0:
                P += w[k] * w[l] * W_cdf[idx - 1]
    return P


def _compute_P(support, weights):
    n = len(support)
    if n <= 56:
        return _compute_P_vectorized(support, weights)
    else:
        return _compute_P_fast(support, weights)


def _compute_P_soft(support, weights, tau=0.05):
    n = len(support)
    w = weights / weights.sum()
    if n <= 1:
        return 0.0
    S = support[:, None] + support[None, :]
    thr = 2.0 * support[None, :] - support[:, None]
    margin = thr - S
    soft = 1.0 / (1.0 + np.exp(-np.clip(margin / tau, -50, 50)))
    W = np.outer(w, w)
    S_4d = S[None, None, :, :]
    soft_4d = soft[:, :, None, None]
    inner = np.einsum('klmn,klmn->kl', soft_4d, W[None, None, :, :])
    P = np.sum(w[:, None] * w[None, :] * inner)
    return float(P)


def _compute_P_grad_soft(support, w, tau=0.05):
    """Compute P_soft and its gradient wrt w (normalized)."""
    n = len(support)
    S = support[:, None] + support[None, :]
    thr = 2.0 * support[None, :] - support[:, None]
    margin = thr - S
    soft = 1.0 / (1.0 + np.exp(-np.clip(margin / tau, -50, 50)))
    # soft' = soft*(1-soft)/tau
    soft_grad = soft * (1.0 - soft) / tau
    W = np.outer(w, w)
    inner = np.einsum('klmn,klmn->kl', soft[:, :, None, None], W[None, None, :, :])
    P = np.sum(w[:, None] * w[None, :] * inner)
    # gradient: dP/dw_m = sum_{k,l} [ w_k w_l d_inner/dw_m + inner*(delta_km w_l + w_k delta_lm) ]
    # d_inner/dw_m = sum_{ij} soft_{kli} W_{mj} + W_{im} soft_{klj} ... but W_{ij}=w_i w_j
    # d_inner/dw_m = w_m * sum_{ij} soft_{kli} (delta_im w_j + w_i delta_jm)
    # = w_m * (sum_j soft_{klmj} w_j + sum_i w_i soft_{klim})
    # This is getting complex. Let's use a simpler approach: finite differences or just
    # return P and use Nelder-Mead.
    return P, None


def _optimize_weights_nm(support, max_time=120.0, seed=42, tol=1e-12):
    n = len(support)
    deadline = time.time() + max_time

    def neg_P_log(w_log):
        if time.time() > deadline:
            return 1e10
        w = np.exp(w_log - w_log.max())
        s = w.sum()
        if s < 1e-15:
            return 1e10
        w = w / s
        return -_compute_P(support, w)

    best_P = -1.0
    best_w = np.ones(n) / n

    x0 = np.zeros(n)
    try:
        result = minimize(neg_P_log, x0, method='Nelder-Mead',
                          options={'maxiter': 8000, 'xatol': 1e-10, 'fatol': tol,
                                   'adaptive': True, 'maxfev': 10000})
        w_opt = np.exp(result.x - result.x.max())
        w_opt = w_opt / w_opt.sum()
        P_opt = -result.fun
        if P_opt > best_P:
            best_P = P_opt
            best_w = w_opt
    except Exception:
        pass

    rng = np.random.default_rng(seed)
    restarts = 20
    for i in range(restarts):
        if time.time() > deadline - 5:
            break
        x0 = rng.normal(0, 1.5, n)
        try:
            result = minimize(neg_P_log, x0, method='Nelder-Mead',
                              options={'maxiter': 4000, 'xatol': 1e-10, 'fatol': tol,
                                       'adaptive': True, 'maxfev': 6000})
            w_opt = np.exp(result.x - result.x.max())
            w_opt = w_opt / w_opt.sum()
            P_opt = -result.fun
            if P_opt > best_P:
                best_P = P_opt
                best_w = w_opt
        except Exception:
            pass

    return best_w, best_P


def _optimize_weights_de(support, max_time=120.0, seed=42):
    n = len(support)
    deadline = time.time() + max_time
    if n < 2:
        return np.ones(n) / max(n, 1), 0.0

    def neg_P_log(w_log):
        if time.time() > deadline:
            return 1e10
        w = np.exp(w_log - w_log.max())
        s = w.sum()
        if s < 1e-15:
            return 1e10
        w = w / s
        return -_compute_P(support, w)

    bounds = [(-6.0, 6.0)] * n
    popsize = max(4, min(15, (2000 // max(n, 1)) + 4))
    try:
        result = differential_evolution(
            neg_P_log, bounds,
            seed=seed, maxiter=300, popsize=popsize, tol=1e-8,
            mutation=(0.3, 0.9), recombination=0.7, polish=True,
            init='sobol')
        w_opt = np.exp(result.x - result.x.max())
        w_opt = w_opt / w_opt.sum()
        P_opt = -result.fun
        return w_opt, P_opt
    except Exception:
        return np.ones(n) / n, -1.0


def _polish_soft(support, w_init, max_time=30.0):
    n = len(support)
    deadline = time.time() + max_time

    def neg_P_soft(w_log):
        if time.time() > deadline:
            return 1e10
        w = np.exp(w_log - w_log.max())
        s = w.sum()
        if s < 1e-15:
            return 1e10
        w = w / s
        return -_compute_P_soft(support, w, tau=0.02)

    w0 = np.maximum(w_init, 1e-9)
    w0 = w0 / w0.sum()
    x0 = np.log(w0)
    best_w = w0.copy()
    best_P = _compute_P(support, w0)
    try:
        result = minimize(neg_P_soft, x0, method='Nelder-Mead',
                          options={'maxiter': 4000, 'xatol': 1e-11, 'fatol': 1e-13,
                                   'adaptive': True, 'maxfev': 5000})
        w_opt = np.exp(result.x - result.x.max())
        w_opt = w_opt / w_opt.sum()
        P_opt = _compute_P(support, w_opt)
        if P_opt > best_P:
            best_P = P_opt
            best_w = w_opt
    except Exception:
        pass
    return best_w, best_P


def _polish_exact(support, w_init, max_time=20.0):
    """High-precision polish using the exact objective with tighter tolerances."""
    n = len(support)
    deadline = time.time() + max_time

    def neg_P_log(w_log):
        if time.time() > deadline:
            return 1e10
        w = np.exp(w_log - w_log.max())
        s = w.sum()
        if s < 1e-15:
            return 1e10
        w = w / s
        return -_compute_P(support, w)

    w0 = np.maximum(w_init, 1e-9)
    w0 = w0 / w0.sum()
    x0 = np.log(w0)
    best_w = w0.copy()
    best_P = _compute_P(support, w0)
    try:
        result = minimize(neg_P_log, x0, method='Nelder-Mead',
                          options={'maxiter': 6000, 'xatol': 1e-13, 'fatol': 1e-15,
                                   'adaptive': True, 'maxfev': 8000})
        w_opt = np.exp(result.x - result.x.max())
        w_opt = w_opt / w_opt.sum()
        P_opt = _compute_P(support, w_opt)
        if P_opt > best_P:
            best_P = P_opt
            best_w = w_opt
    except Exception:
        pass
    return best_w, best_P


def _refine_support(support, max_n=48):
    s = np.sort(np.asarray(support, dtype=np.float64))
    n = len(s)
    if n < 2:
        return s
    mids = (s[:-1] + s[1:]) / 2.0
    new = np.empty(2 * n - 1, dtype=np.float64)
    new[0::2] = s
    new[1::2] = mids
    if len(new) > max_n:
        idx = np.linspace(0, len(new) - 1, max_n).astype(int)
        new = new[idx]
    return np.sort(new)


def _pareto_support(n_tail, shape, scale, w0):
    x_m = scale
    ps = np.linspace(0.02, 0.98, n_tail)
    tail = x_m / (1.0 - ps) ** (1.0 / shape)
    support = np.concatenate([[0.0], tail])
    return support


def _pareto_init_weights(n, w0):
    n_tail = n - 1
    rest = np.linspace(1.0, 0.3, n_tail) ** 1.5
    w = np.concatenate([[w0], rest])
    w = w / w.sum()
    return w


def _optimize_support_and_weights_de(n, max_time=120.0, seed=42):
    """Jointly optimize support and weights using DE in log-space for support."""
    deadline = time.time() + max_time

    def neg_P_params(params):
        if time.time() > deadline:
            return 1e10
        log_s = params[:n]
        log_w = params[n:]
        s = np.exp(log_s - log_s.max())
        w = np.exp(log_w - log_w.max())
        w = w / w.sum()
        s = np.sort(s)
        return -_compute_P(s, w)

    bounds = [(-5.0, 5.0)] * n + [(-5.0, 5.0)] * n
    popsize = max(4, min(12, (1500 // (2 * n)) + 4))
    try:
        result = differential_evolution(
            neg_P_params, bounds,
            seed=seed, maxiter=200, popsize=popsize, tol=1e-8,
            mutation=(0.3, 0.9), recombination=0.7, polish=True,
            init='sobol')
        log_s = result.x[:n]
        log_w = result.x[n:]
        s = np.exp(log_s - log_s.max())
        w = np.exp(log_w - log_w.max())
        w = w / w.sum()
        s = np.sort(s)
        P = -result.fun
        return s, w, P
    except Exception:
        return None, None, -1.0


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    start_time = time.time()
    deadline = start_time + 600

    best_support = np.arange(1, 25, dtype=np.float64)
    best_weights = np.ones(24, dtype=np.float64) / 24
    best_P = _compute_P(best_support, best_weights)

    def _try_candidate(support, w_init, max_opt_time, seed, use_soft=True):
        nonlocal best_P, best_support, best_weights
        if len(support) < 2:
            return
        if np.any(support < -1e-9) or not np.all(np.isfinite(support)):
            return
        if max_opt_time < 8:
            return
        w_init = np.maximum(w_init, 1e-9)
        w_init = w_init / w_init.sum()
        w_opt, P_opt = _optimize_weights_nm(support, max_time=max_opt_time, seed=seed)
        if P_opt > best_P:
            best_P = P_opt
            best_support = support.copy()
            best_weights = w_opt.copy()
        if use_soft and (deadline - time.time()) > 15:
            w_pol, P_pol = _polish_soft(support, best_weights if np.array_equal(best_support, support)
                                        else w_opt,
                                        max_time=min(20.0, (deadline - time.time()) - 12))
            if P_pol > best_P:
                best_P = P_pol
                best_support = support.copy()
                best_weights = w_pol.copy()

    # Strategy 0: Chebyshev node supports
    cheb_configs = [
        (16, 1.0), (20, 1.0), (24, 1.0),
        (16, 5.0), (20, 5.0), (24, 5.0),
        (16, 0.1), (20, 0.1), (24, 0.1),
    ]
    for n, L in cheb_configs:
        if time.time() > deadline - 120:
            break
        roots = np.polynomial.chebyshev.chebroots(n)
        support = 0.5 * L * (roots + 1.0)
        support = np.sort(support)
        if np.any(support < -1e-9) or not np.all(np.isfinite(support)):
            continue
        w_uniform = np.ones(n) / n
        P_uniform = _compute_P(support, w_uniform)
        if P_uniform < best_P * 0.3:
            continue
        remaining = deadline - time.time() - 10
        if remaining < 10:
            break
        _try_candidate(support, w_uniform, min(55.0, remaining * 0.5), 42)

    # Strategy 1: geometric supports
    configs = [
        (16, 0.01, 2.0), (20, 0.01, 2.0), (24, 0.01, 2.0),
        (28, 0.01, 2.0), (32, 0.01, 2.0),
        (16, 0.01, 3.0), (20, 0.01, 3.0), (24, 0.01, 3.0),
        (28, 0.01, 3.0),
        (16, 0.1, 2.0), (20, 0.1, 2.0), (24, 0.1, 2.0),
        (20, 0.5, 2.0), (24, 0.5, 2.0),
        (16, 1.0, 2.0), (20, 1.0, 2.0), (24, 1.0, 2.0),
        (16, 0.001, 5.0), (20, 0.001, 5.0), (24, 0.001, 5.0),
        (16, 0.01, 4.0), (20, 0.01, 4.0), (24, 0.01, 4.0),
        (16, 0.5, 3.0), (20, 0.5, 3.0), (24, 0.5, 3.0),
        (16, 0.001, 10.0), (20, 0.001, 10.0),
        (16, 0.0001, 20.0), (20, 0.0001, 20.0),
        (16, 0.0, 2.0), (20, 0.0, 2.0), (24, 0.0, 2.0),
    ]
    for n, a, r in configs:
        if time.time() > deadline - 120:
            break
        if r <= 1.0:
            continue
        if a == 0.0:
            support = np.concatenate([np.zeros(1), r ** np.arange(1, n, dtype=np.float64)])
        else:
            support = a * (r ** np.arange(n, dtype=np.float64))
        if np.any(support < 0) or not np.all(np.isfinite(support)):
            continue
        w_uniform = np.ones(n) / n
        P_uniform = _compute_P(support, w_uniform)
        if P_uniform < best_P * 0.3:
            continue
        remaining = deadline - time.time() - 10
        if remaining < 10:
            break
        _try_candidate(support, w_uniform, min(55.0, remaining * 0.5), 42)

    # Strategy 2: bimodal support
    for n_tail, a, r, w0 in [
        (16, 0.01, 3.0, 0.3), (20, 0.01, 3.0, 0.3), (24, 0.01, 3.0, 0.3),
        (16, 0.05, 3.0, 0.4), (20, 0.05, 3.0, 0.4), (24, 0.05, 3.0, 0.4),
        (16, 0.1, 2.5, 0.3), (20, 0.1, 2.5, 0.3), (24, 0.1, 2.5, 0.3),
        (20, 0.01, 4.0, 0.2), (24, 0.01, 4.0, 0.2),
    ]:
        if time.time() > deadline - 90:
            break
        n = n_tail + 1
        tail = a * (r ** np.arange(n_tail, dtype=np.float64))
        support = np.concatenate([[0.0], tail])
        if np.any(support < 0) or not np.all(np.isfinite(support)):
            continue
        w_init = np.ones(n) / n
        w_init[0] = w0
        w_init = w_init / w_init.sum()
        P_init = _compute_P(support, w_init)
        if P_init < best_P * 0.4:
            continue
        remaining = deadline - time.time() - 10
        if remaining < 10:
            break
        _try_candidate(support, w_init, min(65.0, remaining * 0.5), 7)

    # Strategy 3: Pareto tail
    for n_tail, shape, scale, w0 in [
        (16, 1.5, 1.0, 0.30), (20, 1.5, 1.0, 0.30), (24, 1.5, 1.0, 0.30),
        (16, 2.0, 1.0, 0.40), (20, 2.0, 1.0, 0.40), (24, 2.0, 1.0, 0.40),
        (16, 1.0, 1.0, 0.25), (20, 1.0, 1.0, 0.25), (24, 1.0, 1.0, 0.25),
        (16, 3.0, 1.0, 0.50), (20, 3.0, 1.0, 0.50),
        (16, 1.5, 0.1, 0.35), (20, 1.5, 0.1, 0.35), (24, 1.5, 0.1, 0.35),
        (16, 2.0, 5.0, 0.45), (20, 2.0, 5.0, 0.45),
        (16, 1.2, 1.0, 0.30), (20, 1.2, 1.0, 0.30), (24, 1.2, 1.0, 0.30),
    ]:
        if time.time() > deadline - 90:
            break
        n = n_tail + 1
        support = _pareto_support(n_tail, shape, scale, w0)
        if np.any(support < -1e-9) or not np.all(np.isfinite(support)):
            continue
        w_init = _pareto_init_weights(n, w0)
        P_init = _compute_P(support, w_init)
        if P_init < best_P * 0.4:
            continue
        remaining = deadline - time.time() - 10
        if remaining < 10:
            break
        _try_candidate(support, w_init, min(65.0, remaining * 0.5), 13)

    # Strategy 4: coarse-to-fine refinement + DE
    if time.time() < deadline - 60:
        for refine_round in range(3):
            if time.time() > deadline - 40:
                break
            refined = _refine_support(best_support, max_n=48)
            if len(refined) < 2 or len(refined) == len(best_support):
                continue
            n_r = len(refined)
            if len(best_weights) == len(best_support):
                w_interp = np.interp(refined, best_support, best_weights)
                w_interp = np.maximum(w_interp, 1e-6)
                w_interp = w_interp / w_interp.sum()
            else:
                w_interp = np.ones(n_r) / n_r
            remaining = deadline - time.time() - 10
            if remaining < 15:
                break
            w_opt, P_opt = _optimize_weights_de(refined, max_time=min(55.0, remaining * 0.6),
                                                seed=11 + refine_round)
            if P_opt > best_P:
                best_P = P_opt
                best_support = refined.copy()
                best_weights = w_opt.copy()
            if time.time() < deadline - 25:
                w_opt2, P_opt2 = _optimize_weights_nm(refined,
                                                      max_time=min(30.0, (deadline - time.time() - 20) * 0.5),
                                                      seed=23)
                if P_opt2 > best_P:
                    best_P = P_opt2
                    best_support = refined.copy()
                    best_weights = w_opt2.copy()

    # Strategy 5: scale search
    if time.time() < deadline - 30:
        base = best_support.copy()
        for scale in [0.2, 0.5, 2.0, 5.0]:
            if time.time() > deadline - 20:
                break
            cand = base * scale
            if np.any(cand < 0) or not np.all(np.isfinite(cand)):
                continue
            w = best_weights.copy()
            P = _compute_P(cand, w)
            if P > best_P:
                best_P = P
                best_support = cand.copy()
                best_weights = w.copy()

    # Strategy 6 (NEW): joint support+weights DE on small n
    if time.time() < deadline - 50:
        for n_joint in [8, 12, 16]:
            if time.time() > deadline - 30:
                break
            remaining = deadline - time.time() - 10
            if remaining < 20:
                break
            s_opt, w_opt, P_opt = _optimize_support_and_weights_de(
                n_joint, max_time=min(40.0, remaining * 0.6), seed=99)
            if s_opt is not None and P_opt > best_P:
                if np.all(s_opt >= 0) and np.all(np.isfinite(s_opt)) and np.all(np.isfinite(w_opt)):
                    best_P = P_opt
                    best_support = s_opt.copy()
                    best_weights = w_opt.copy()

    # Final polish: soft then exact
    if time.time() < deadline - 8 and len(best_support) >= 2:
        w_pol, P_pol = _polish_soft(best_support, best_weights,
                                    max_time=min(20.0, (deadline - time.time()) - 12))
        if P_pol > best_P:
            best_P = P_pol
            best_weights = w_pol.copy()
        if time.time() < deadline - 5:
            w_pol2, P_pol2 = _polish_exact(best_support, best_weights,
                                           max_time=min(15.0, (deadline - time.time()) - 5))
            if P_pol2 > best_P:
                best_P = P_pol2
                best_weights = w_pol2.copy()

    if time.time() > deadline - 5:
        return best_support.astype(np.float64), best_weights.astype(np.float64)

    if len(best_support) != len(best_weights):
        return np.arange(1, 25, dtype=np.float64), np.ones(24, dtype=np.float64) / 24
    if np.any(best_support < -1e-9) or np.any(best_weights < -1e-9):
        return np.arange(1, 25, dtype=np.float64), np.ones(24, dtype=np.float64) / 24
    if not np.all(np.isfinite(best_support)) or not np.all(np.isfinite(best_weights)):
        return np.arange(1, 25, dtype=np.float64), np.ones(24, dtype=np.float64) / 24
    if best_weights.sum() <= 1e-9:
        return np.arange(1, 25, dtype=np.float64), np.ones(24, dtype=np.float64) / 24

    best_support = np.maximum(best_support, 0.0)
    best_weights = np.maximum(best_weights, 0.0)

    return best_support.astype(np.float64), best_weights.astype(np.float64)


# EVOLVE-BLOCK-END
