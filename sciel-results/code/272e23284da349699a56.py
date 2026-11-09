# EVOLVE-BLOCK-START
import time
import numpy as np

_N = 20
_DEADLINE = time.time() + 580.0  # safety margin before the 630 s wall-clock kill


def _normalize(pts):
    """Subtract centroid and divide by mean distance to centroid."""
    c = pts.mean(axis=0)
    p = pts - c
    md = np.linalg.norm(p, axis=1).mean()
    if md <= 1e-12:
        return p, md
    return p / md, md


def _turn_signs(p):
    """Cross products of consecutive edges (cyclic)."""
    e1 = np.roll(p, -1, axis=0) - p
    e2 = np.roll(p, -2, axis=0) - np.roll(p, -1, axis=0)
    return e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]


def _is_valid(p):
    if not np.all(np.isfinite(p)):
        return False
    if p.shape != (_N, 2):
        return False
    # no repeated points
    d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    if d.min() < 1e-6:
        return False
    cr = _turn_signs(p)
    if np.any(np.abs(cr) < 1e-9):
        return False
    if not (np.all(cr > 0) or np.all(cr < 0)):
        return False
    return True


def _vertex_scores(p):
    """Per-vertex equidistance scores in [0,1] (0 if any hard failure)."""
    n = p.shape[0]
    d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=2)
    scores = np.zeros(n)
    for i in range(n):
        row = d[i].copy()
        row[i] = np.inf
        ds = np.sort(row)
        if ds[0] < 0.05:
            continue
        diffs = np.diff(ds)
        k = int(np.argmin(diffs))
        D = 0.5 * (ds[k] + ds[k + 1])
        if D < 0.1:
            continue
        if k == 0:
            sel = ds[0:4]
        elif k == n - 3:  # k == 17 for n=20
            sel = ds[n - 5:n - 1]
        else:
            sel = ds[k - 1:k + 3]
        raw = np.mean(np.maximum(sel / D, D / sel))
        scores[i] = min(1.0, 1.0 / raw) if raw > 0 else 0.0
    return scores


def _objective_flat(flat):
    """Smooth surrogate: -softmin of per-vertex scores (minimize)."""
    p = flat.reshape(_N, 2)
    p, md = _normalize(p)
    if md <= 0.01:
        return 1e6
    sc = _vertex_scores(p)
    # soft-min with large temperature -> approaches true min
    m = sc.min()
    s = m - np.logaddexp.reduce(sc - m)
    # add a tiny convexity regularizer to discourage degenerate turns
    cr = _turn_signs(p)
    reg = 1e-4 * np.sum(np.log1p(np.abs(cr)))
    return -s + reg


def _best_valid_from(starts, base):
    best_pts, best_metric = base, -1.0
    for s in starts:
        p, _ = _normalize(s)
        if not _is_valid(p):
            continue
        sc = _vertex_scores(p)
        m = sc.min()
        if m > best_metric:
            best_metric = m
            best_pts = p
    return best_pts, best_metric


def equidistant_points_polygon20() -> np.ndarray:
    from scipy.optimize import minimize

    # ---- deterministic regular polygon baseline (score 1.0) ----
    angles = np.linspace(0.0, 2.0 * np.pi, _N, endpoint=False)
    base = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    base, _ = _normalize(base)
    best_pts, best_metric = base, float(_vertex_scores(base).min())

    # ---- build diverse initial points ----
    starts = [base.copy()]
    # radial perturbation: r(theta) = 1 + a1*cos + b1*sin + a2*cos2 + b2*sin2
    rng = np.random.default_rng(12345)
    for _ in range(14):
        a1 = rng.uniform(-0.25, 0.25)
        b1 = rng.uniform(-0.25, 0.25)
        a2 = rng.uniform(-0.15, 0.15)
        b2 = rng.uniform(-0.15, 0.15)
        th = angles
        r = 1.0 + a1 * np.cos(th) + b1 * np.sin(th) + a2 * np.cos(2 * th) + b2 * np.sin(2 * th)
        r = np.clip(r, 0.6, 1.4)
        pts = np.stack([r * np.cos(th), r * np.sin(th)], axis=1)
        # ensure strictly convex ordering by sorting around centroid
        c = pts.mean(axis=0)
        ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
        order = np.argsort(ang)
        pts = pts[order]
        starts.append(pts)
    # a couple of pure-noise starts
    for _ in range(2):
        pts = np.stack([np.cos(angles), np.sin(angles)], axis=1) + 0.05 * rng.standard_normal((_N, 2))
        starts.append(pts)

    # ---- multi-restart L-BFGS-B on the smooth surrogate ----
    for s in starts:
        if time.time() > _DEADLINE - 40.0:
            break
        p0, _ = _normalize(s)
        try:
            res = minimize(
                _objective_flat,
                p0.ravel(),
                method="L-BFGS-B",
                options={"maxiter": 800, "maxfun": 20000, "gtol": 1e-8},
            )
        except Exception:
            continue
        p, _ = _normalize(res.x.reshape(_N, 2))
        if not _is_valid(p):
            continue
        sc = _vertex_scores(p)
        m = sc.min()
        if m > best_metric:
            best_metric = m
            best_pts = p

    # ---- high-precision polish on the best found (tighter tol, more iters) ----
    if time.time() < _DEADLINE - 30.0:
        try:
            res = minimize(
                _objective_flat,
                best_pts.ravel(),
                method="L-BFGS-B",
                options={"maxiter": 3000, "maxfun": 200000, "gtol": 1e-12},
            )
            p, _ = _normalize(res.x.reshape(_N, 2))
            if _is_valid(p):
                sc = _vertex_scores(p)
                m = sc.min()
                if m > best_metric:
                    best_metric = m
                    best_pts = p
        except Exception:
            pass

    # final validity guard; fall back to the regular polygon
    if not _is_valid(best_pts):
        best_pts = base
    return best_pts


# EVOLVE-BLOCK-END
