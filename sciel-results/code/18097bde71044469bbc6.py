# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, minimize_scalar


def _compute_P_fast(support, weights):
    """Exact P[X1+X2+X3 < 2*X4] using sorted pair-sums + searchsorted. O(n^2 log n)."""
    s = np.asarray(support, dtype=np.float64).ravel()
    w = np.asarray(weights, dtype=np.float64).ravel()
    n = len(s)
    if n == 0:
        return 0.0
    total = w.sum()
    if not np.isfinite(total) or total <= 1e-9:
        return 0.0
    p = w / total

    pair_sums = s[:, None] + s[None, :]
    pair_probs = p[:, None] * p[None, :]

    flat_sums = pair_sums.ravel()
    flat_probs = pair_probs.ravel()
    order = np.argsort(flat_sums, kind='mergesort')
    sorted_sums = flat_sums[order]
    sorted_probs = flat_probs[order]
    cumsum_probs = np.cumsum(sorted_probs)

    thresholds = 2.0 * s[None, :] - s[:, None]
    flat_thresholds = thresholds.ravel()

    counts = np.searchsorted(sorted_sums, flat_thresholds, side='left')
    F = np.zeros_like(flat_thresholds)
    mask = counts > 0
    F[mask] = cumsum_probs[counts[mask] - 1]
    F = F.reshape(n, n)

    weight_matrix = p[:, None] * p[None, :]
    P = np.sum(weight_matrix * F)
    return float(P)


def _two_point_p(p):
    """P for {0, 1} with P(0)=p, P(1)=1-p. = (1-p)*p^2*(3-2p)."""
    q = 1.0 - p
    return q * p * p * (3.0 - 2.0 * p)


def _optimize_two_point():
    res = minimize_scalar(lambda p: -_two_point_p(p), bounds=(1e-6, 1.0 - 1e-6), method='bounded')
    p_opt = float(res.x)
    return p_opt, _two_point_p(p_opt)


def _optimize_weights_fixed_support(support, initial_weights, deadline, max_iter=800):
    """Refine weights (simplex) for a FIXED support: coordinate ascent + log-simplex NM."""
    s = np.asarray(support, dtype=np.float64).ravel()
    n = len(s)
    if n < 1:
        return np.asarray(initial_weights), 0.0
    w = np.asarray(initial_weights, dtype=np.float64).ravel().copy()
    w = np.maximum(w, 0.0)
    tot = w.sum()
    if tot <= 1e-9:
        w = np.full(n, 1.0 / n)
        tot = 1.0
    w = w / tot

    best_P = _compute_P_fast(s, w)
    best_w = w.copy()

    # --- coordinate ascent on the simplex ---
    step = 0.05
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        improved = False
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                for delta in (step, -step):
                    nw = w.copy()
                    nw[i] -= delta
                    nw[j] += delta
                    if nw[i] < 1e-9 or nw[j] < 1e-9:
                        continue
                    P = _compute_P_fast(s, nw)
                    if P > best_P + 1e-12:
                        best_P = P
                        best_w = nw.copy()
                        w = nw.copy()
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            step *= 0.5
            if step < 1e-7:
                break

    # --- log-simplex Nelder-Mead refinement ---
    if time.time() < deadline - 15 and n >= 2:
        def neg_P_log(log_w):
            ew = np.exp(log_w - log_w.max())
            ew = ew / ew.sum()
            return -_compute_P_fast(s, ew)

        w_init = np.maximum(best_w, 1e-12)
        log_init = np.log(w_init)
        log_init = log_init - log_init.max()
        try:
            r = minimize(neg_P_log, log_init, method='Nelder-Mead',
                         options={'maxiter': 500, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True})
            ew = np.exp(r.x - r.x.max())
            ew = ew / ew.sum()
            P = _compute_P_fast(s, ew)
            if P > best_P:
                best_P = P
                best_w = ew.copy()
        except Exception:
            pass
    return best_w, float(best_P)


def _make_parametric_support(params, n_points):
    """
    Create a parametric support from parameters.
    params = [log(a1), log(a2), ..., log(a_{n-1}), log_weight_ratio]
    Support: [0, a1, a1*a2, a1*a2*a3, ..., prod(a_i)]
    Weights: geometric with ratio determined by last param.
    """
    if n_points <= 1:
        return np.array([0.0]), np.array([1.0])

    ratios = np.exp(params[:n_points - 1])
    # Clip ratios to reasonable range
    ratios = np.clip(ratios, 0.1, 100.0)

    support = np.zeros(n_points)
    support[0] = 0.0
    for i in range(1, n_points):
        support[i] = support[i - 1] * ratios[i - 1] if i > 1 else ratios[0]
        if i == 1:
            support[i] = ratios[0]
        else:
            support[i] = support[i - 1] * ratios[i - 1]

    # Ensure sorted and positive
    support = np.sort(support)
    support = np.maximum(support, 0.0)

    # Weights: geometric decay
    log_wr = params[n_points - 1]
    wr = np.exp(np.clip(log_wr, -5.0, 5.0))
    wr = np.clip(wr, 0.01, 100.0)

    weights = wr ** np.arange(n_points)
    weights = weights / weights.sum()

    return support, weights


def _parametric_objective(params, n_points, deadline):
    """Negative P for parametric support."""
    try:
        if time.time() > deadline:
            return 1e10
        support, weights = _make_parametric_support(params, n_points)
        if len(support) != n_points:
            return 1e10
        if not np.all(np.isfinite(support)) or not np.all(np.isfinite(weights)):
            return 1e10
        if np.any(support < -1e-9) or np.any(weights < -1e-9):
            return 1e10
        if weights.sum() <= 1e-9:
            return 1e10
        P = _compute_P_fast(support, weights)
        return -P
    except Exception:
        return 1e10


def _optimize_parametric(n_points, deadline, initial_params=None):
    """Optimize parametric support and weights jointly."""
    dim = n_points  # n-1 ratios + 1 weight ratio
    if initial_params is None:
        # Initialize with reasonable values
        initial_params = np.zeros(dim)
        # Set ratios to log(2) initially
        initial_params[:n_points - 1] = np.log(2.0)
        # Set weight ratio to log(0.5)
        initial_params[n_points - 1] = np.log(0.5)

    best_P = -1e10
    best_support = None
    best_weights = None

    # Try multiple random starts
    n_starts = max(2, int((deadline - time.time()) / 30))
    n_starts = min(n_starts, 5)

    for start_i in range(n_starts):
        if time.time() > deadline - 20:
            break

        if start_i == 0:
            params = initial_params.copy()
        else:
            params = initial_params + np.random.normal(0, 0.5, dim)
            params[:n_points - 1] = np.clip(params[:n_points - 1], -3, 5)
            params[n_points - 1] = np.clip(params[n_points - 1], -5, 5)

        try:
            r = minimize(_parametric_objective, params, args=(n_points, deadline),
                         method='Nelder-Mead',
                         options={'maxiter': 300, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True})
            if r.fun < 0:  # Valid result
                P = -r.fun
                support, weights = _make_parametric_support(r.x, n_points)
                if P > best_P:
                    best_P = P
                    best_support = support.copy()
                    best_weights = weights.copy()
        except Exception:
            pass

    # If we found a good parametric solution, refine weights
    if best_support is not None and best_P > -1e9 and time.time() < deadline - 10:
        refined_w, refined_P = _optimize_weights_fixed_support(best_support, best_weights, deadline)
        if refined_P > best_P:
            best_P = refined_P
            best_weights = refined_w

    return best_support, best_weights, best_P


def _try_geometric_support(ratio, n_points, deadline):
    if ratio <= 1.0:
        return None, None, 0.0
    support = np.array([0.0] + [ratio**i for i in range(1, n_points)], dtype=np.float64)
    weights = np.array([0.5 * (0.5 ** i) for i in range(n_points)], dtype=np.float64)
    weights = weights / weights.sum()
    best_w, best_P = _optimize_weights_fixed_support(support, weights, deadline)
    return support, best_w, best_P


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    start_time = time.time()
    deadline = start_time + 600.0

    best_P = -1.0
    best_support = None
    best_weights = None

    def _update(sup, wts, P):
        nonlocal best_P, best_support, best_weights
        if P > best_P:
            best_P = P
            best_support = np.asarray(sup, dtype=np.float64).ravel().copy()
            best_weights = np.asarray(wts, dtype=np.float64).ravel().copy()

    # ---- Phase 1: two-point analytic optimum ----
    p_opt, P_two = _optimize_two_point()
    _update(np.array([0.0, 1.0], dtype=np.float64),
            np.array([p_opt, 1.0 - p_opt], dtype=np.float64), P_two)

    if time.time() > deadline:
        return best_support, best_weights

    # ---- Phase 2: Parametric heavy-tail optimization (key new strategy) ----
    # Try different numbers of points with parametric optimization
    for n_points in [3, 4, 5, 6, 7, 8]:
        if time.time() > deadline - 60:
            break
        sup, wts, P = _optimize_parametric(n_points, deadline)
        if sup is not None and P > best_P:
            _update(sup, wts, P)

    if time.time() > deadline:
        return best_support, best_weights

    # ---- Phase 3: Geometric supports with optimized weights ----
    for ratio in [1.5, 2.0, 3.0, 4.0, 5.0, 10.0]:
        for n_points in [3, 4, 5, 6, 7, 8]:
            if time.time() > deadline - 30:
                break
            sup, wts, P = _try_geometric_support(ratio, n_points, deadline)
            if sup is not None and P > best_P:
                _update(sup, wts, P)

    if time.time() > deadline:
        return best_support, best_weights

    # ---- Phase 4: Add points to the best support ----
    if best_support is not None and time.time() < deadline - 25:
        sup = best_support.copy()
        wts = best_weights.copy()
        max_val = float(sup.max())
        if max_val <= 0:
            max_val = 1.0
        candidates = sorted(set([0.0, max_val * 0.5, max_val * 2.0, max_val * 5.0,
                                 max_val * 0.25, max_val * 10.0]))
        for new_pos in candidates:
            if time.time() > deadline - 25:
                break
            if np.any(np.abs(sup - new_pos) < 1e-9 * max(1.0, max_val)):
                continue
            new_sup = np.sort(np.append(sup, new_pos))
            # Rebuild weights
            old_with_new = np.append(sup, new_pos)
            order = np.argsort(old_with_new)
            new_wts = np.append(wts, 0.0)[order]
            for eps in [0.05, 0.1, 0.2, 0.3]:
                if time.time() > deadline - 25:
                    break
                cand_w = new_wts * (1.0 - eps)
                cand_w[-1] = eps
                cand_w = np.maximum(cand_w, 0.0)
                P = _compute_P_fast(new_sup, cand_w)
                if P > best_P:
                    opt_w, P2 = _optimize_weights_fixed_support(new_sup, cand_w, deadline)
                    if P2 > best_P:
                        _update(new_sup, opt_w, P2)
                    sup = best_support.copy()
                    wts = best_weights.copy()
                    max_val = float(sup.max())

    # ---- Phase 5: Final weight polish ----
    if best_support is not None and time.time() < deadline - 5:
        opt_w, P2 = _optimize_weights_fixed_support(best_support, best_weights, deadline)
        if P2 > best_P:
            _update(best_support, opt_w, P2)

    # ---- validate / fallback ----
    if best_support is None or best_weights is None:
        n = 32
        best_support = np.arange(n, dtype=np.float64)
        best_weights = np.ones(n, dtype=np.float64) / n
    else:
        best_support = np.asarray(best_support, dtype=np.float64).ravel()
        best_weights = np.asarray(best_weights, dtype=np.float64).ravel()
        mask = (best_support >= -1e-9) & (best_weights >= -1e-9) & \
               np.isfinite(best_support) & np.isfinite(best_weights)
        best_support = best_support[mask]
        best_weights = best_weights[mask]
        best_support = np.maximum(best_support, 0.0)
        best_weights = np.maximum(best_weights, 0.0)
        if len(best_support) == 0 or best_weights.sum() <= 1e-9:
            n = 32
            best_support = np.arange(n, dtype=np.float64)
            best_weights = np.ones(n, dtype=np.float64) / n

    return best_support, best_weights


# EVOLVE-BLOCK-END
