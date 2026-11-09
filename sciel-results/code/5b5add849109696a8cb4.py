# EVOLVE-BLOCK-START
import time
import numpy as np

_START = time.perf_counter()
_DEADLINE = 620.0  # safety margin before 630s kill

# Parameterization: regular 20-gon on unit circle, with low-frequency Fourier
# deviations in both angle (theta) and radius (r). This keeps the polygon smooth
# and globally convex while giving the optimizer a low-dimensional manifold.
_MODES = (1, 2, 3, 4)


def _build_polygon(params):
    """params: array of length 8*M. First 8*M//2 are angle coeffs, rest radius coeffs.
    Layout: for each mode k, angle: (a_cos, a_sin), radius: (r_cos, r_sin)."""
    n = 20
    j = np.arange(n)
    base_ang = 2.0 * np.pi * j / n
    M = len(_MODES)
    npar = 4 * M
    a = params[:2 * M]
    r = params[2 * M:]
    ang = base_ang.copy()
    rad = np.ones(n)
    for i, k in enumerate(_MODES):
        ang += a[2 * i] * np.cos(k * base_ang) + a[2 * i + 1] * np.sin(k * base_ang)
        rad += r[2 * i] * np.cos(k * base_ang) + r[2 * i + 1] * np.sin(k * base_ang)
    # keep radius positive
    rad = np.clip(rad, 0.2, 3.0)
    x = rad * np.cos(ang)
    y = rad * np.sin(ang)
    return np.stack([x, y], axis=1)


def _vertex_scores(poly):
    """Return per-vertex score (20,) per the contract."""
    n = poly.shape[0]
    d2 = np.sum((poly[:, None, :] - poly[None, :, :]) ** 2, axis=2)
    d = np.sqrt(d2)
    np.fill_diagonal(d, np.inf)
    scores = np.zeros(n)
    for v in range(n):
        dv = np.sort(d[v])
        dv = dv[:n - 1]
        if np.any(dv < 0.05):
            scores[v] = 0.0
            continue
        gaps = dv[1:] - dv[:-1]
        k = int(np.argmin(gaps))
        D = 0.5 * (dv[k] + dv[k + 1])
        if D < 0.1:
            scores[v] = 0.0
            continue
        if k == 0:
            sel = dv[0:4]
        elif k == 17:
            sel = dv[15:19]
        else:
            sel = dv[k - 1:k + 3]
        ratios = np.maximum(sel / D, D / sel)
        raw = np.mean(ratios)
        scores[v] = min(1.0, 1.0 / raw)
    return scores


def _smooth_min(scores, temp):
    """Soft-min via log-sum-exp on -scores."""
    if temp <= 0:
        return float(np.min(scores))
    neg = -scores
    m = np.max(neg)
    return float(m + temp * np.log(np.sum(np.exp(neg - m))))


def _objective(params, temp):
    """Minimize: -softmin(scores) + convexity penalty."""
    poly = _build_polygon(params)
    # convexity / turn-sign check (penalize violations)
    n = poly.shape[0]
    turns = np.zeros(n)
    for i in range(n):
        p0 = poly[i]
        p1 = poly[(i + 1) % n]
        p2 = poly[(i + 2) % n]
        e1 = p1 - p0
        e2 = p2 - p1
        cross = e1[0] * e2[1] - e1[1] * e2[0]
        turns[i] = cross
    # want all same sign and nonzero
    mean_t = np.mean(turns)
    sign = 1.0 if mean_t >= 0 else -1.0
    turn_penalty = np.sum(np.minimum(0.0, sign * turns) ** 2) * 100.0
    # repeated points penalty
    min_dist = np.min(d2 := np.sum((poly[:, None, :] - poly[None, :, :]) ** 2, axis=2)
                      , axis=1) if False else None
    # compute min pairwise distance properly
    pd2 = np.sum((poly[:, None, :] - poly[None, :, :]) ** 2, axis=2)
    np.fill_diagonal(pd2, np.inf)
    rep = np.min(pd2)
    rep_penalty = np.maximum(0.0, 0.05 - np.sqrt(rep)) ** 2 * 1000.0

    scores = _vertex_scores(poly)
    sm = _smooth_min(scores, temp)
    return -(sm) + turn_penalty + rep_penalty, scores


def _finite_grad_fmin(x0, fun, fprime, maxiter=200, xtol=1e-8, ftol=1e-10):
    """Simple gradient descent with backtracking, deadline-aware."""
    x = np.array(x0, dtype=float)
    f, _ = fun(x)
    for it in range(maxiter):
        if time.perf_counter() - _START > _DEADLINE:
            break
        g, _ = fprime(x)
        # backtracking line search
        step = 0.5
        fx = f
        for _ in range(30):
            xn = x - step * g
            fn, _ = fun(xn)
            if fn < fx - 1e-12 * abs(fx):
                break
            step *= 0.5
        else:
            break
        if np.linalg.norm(step * g) < xtol:
            break
        x = xn
        f = fn
        if abs(fx - f) < ftol:
            break
    return x, f


def _num_grad(params, temp):
    eps = 1e-6
    g = np.zeros_like(params)
    f0, _ = _objective(params, temp)
    for i in range(len(params)):
        pp = params.copy()
        pp[i] += eps
        fp, _ = _objective(pp, temp)
        pm = params.copy()
        pm[i] -= eps
        fm, _ = _objective(pm, temp)
        g[i] = (fp - fm) / (2 * eps)
    return g, f0


def equidistant_points_polygon20() -> np.ndarray:
    n = 20
    M = len(_MODES)
    npar = 4 * M

    # best-so-far tracking
    base_poly = None
    base_score = -1.0

    def record(poly):
        nonlocal base_poly, base_score
        s = _vertex_scores(poly)
        # validity check
        n2 = poly.shape[0]
        turns = []
        for i in range(n2):
            e1 = poly[(i + 1) % n2] - poly[i]
            e2 = poly[(i + 2) % n2] - poly[(i + 1) % n2]
            turns.append(e1[0] * e2[1] - e1[1] * e2[0])
        turns = np.array(turns)
        if np.any(np.abs(turns) < 1e-12):
            return
        if not (np.all(turns > 0) or np.all(turns < 0)):
            return
        # no repeated points
        pd2 = np.sum((poly[:, None, :] - poly[None, :, :]) ** 2, axis=2)
        np.fill_diagonal(pd2, np.inf)
        if np.min(pd2) < (0.05 ** 2):
            return
        # centroid normalization check (mean dist > 0.01)
        c = np.mean(poly, axis=0)
        pc = poly - c
        md = np.mean(np.sqrt(np.sum(pc ** 2, axis=1)))
        if md <= 0.01:
            return
        mn = float(np.min(s))
        if mn > base_score:
            base_score = mn
            base_poly = poly.copy()

    # start: regular polygon (params = 0)
    p0 = np.zeros(npar)
    poly0 = _build_polygon(p0)
    record(poly0)

    # anneal temperature from coarse to fine
    temps = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002]
    cur = p0.copy()
    for temp in temps:
        if time.perf_counter() - _START > _DEADLINE - 5.0:
            break
        try:
            cur, f = _finite_grad_fmin(cur, lambda p: _objective(p, temp),
                                       lambda p: _num_grad(p, temp),
                                       maxiter=150)
        except Exception:
            break
        poly = _build_polygon(cur)
        record(poly)

    # final polish: direct minimization of the hard min with tiny steps
    if time.perf_counter() - _START < _DEADLINE - 3.0:
        # a few local perturbations to escape plateaus
        best = cur.copy()
        bs = base_score
        for _ in range(40):
            if time.perf_counter() - _START > _DEADLINE - 2.0:
                break
            cand = best + 0.02 * np.random.randn(npar)
            poly = _build_polygon(cand)
            s = _vertex_scores(poly)
            mn = float(np.min(s))
            if mn > bs + 1e-9:
                best = cand
                bs = mn
                record(poly)

    if base_poly is None:
        # fallback to regular polygon
        angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        base_poly = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    return base_poly


# EVOLVE-BLOCK-END
