"""Heilbronn convex regions: maximize min triangle area / convex hull area for 13 points.

Combined strategy:
- Structural idea from inspiration 1: exploit the strong two-ring (regular polygon + regular
  polygon) parameterization as a low-dimensional manifold where the optimum likely lives.
  Optimize the 4 parameters (r_hull, r_inner, rot_hull, rot_inner) with a smooth soft-min
  surrogate (log-sum-exp) via L-BFGS-B, then polish with the hard objective via Nelder-Mead.
- Refinement machinery from the current program: fast vectorized triangle-area computation,
  targeted 'lift' moves on the smallest triangle, and a deterministic local polish that
  iteratively improves the worst triangle.  These are applied to the free-space (13x2)
  configuration produced by the symmetric optimizer to break residual asymmetry and squeeze
  out the last bits of score.
"""
import time
import numpy as np
import scipy.optimize as opt
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13
_DEADLINE = time.time() + 145.0  # wall-clock guard (150 s kill, 5 s safety margin)


def _time_ok(margin=0.0):
    return time.time() < _DEADLINE - margin


# Precompute triple indices once
_TRIPLES = list(combinations(range(N), 3))
_I_IDX = np.array([t[0] for t in _TRIPLES])
_J_IDX = np.array([t[1] for t in _TRIPLES])
_K_IDX = np.array([t[2] for t in _TRIPLES])


def _all_areas(pts):
    """Vectorized triangle areas for all C(n,3) triples."""
    x = pts[:, 0]
    y = pts[:, 1]
    v1x = x[_J_IDX] - x[_I_IDX]
    v1y = y[_J_IDX] - y[_I_IDX]
    v2x = x[_K_IDX] - x[_I_IDX]
    v2y = y[_K_IDX] - y[_I_IDX]
    return 0.5 * np.abs(v1x * v2y - v1y * v2x)


def _hull_area(pts):
    try:
        return ConvexHull(pts).volume
    except Exception:
        return 0.0


def _score(pts):
    hull_a = _hull_area(pts)
    if hull_a <= 1e-12:
        return 0.0
    areas = _all_areas(pts)
    if areas.size == 0:
        return 0.0
    return float(np.min(areas)) / hull_a


def _soft_min_score(pts, beta=50.0):
    """Return negative smooth surrogate of min-area / hull-area (to minimize)."""
    hull_a = _hull_area(pts)
    if hull_a <= 1e-12:
        return 1e6
    areas = _all_areas(pts)
    neg_beta = -beta * areas
    mx = np.max(neg_beta)
    if mx < -1e6:
        return 1e6
    log_sum_exp = mx + np.log(np.sum(np.exp(neg_beta - mx)))
    soft_min = -log_sum_exp / beta
    return -(soft_min / hull_a)


def _soft_objective(flat, beta=50.0):
    pts = flat.reshape(-1, 2)
    return _soft_min_score(pts, beta)


def _make_symmetric(n_hull, n_inner, r_hull, r_inner, rot_hull, rot_inner):
    """Build a two-ring regular-polygon configuration."""
    pts = []
    if n_hull > 0:
        ang = rot_hull + 2 * np.pi * np.arange(n_hull) / n_hull
        pts.append(r_hull * np.column_stack([np.cos(ang), np.sin(ang)]))
    if n_inner > 0:
        ang = rot_inner + 2 * np.pi * np.arange(n_inner) / n_inner
        pts.append(r_inner * np.column_stack([np.cos(ang), np.sin(ang)]))
    return np.vstack(pts)


def _symmetric_objective(params, n_hull, n_inner, beta=50.0):
    r_hull, r_inner, rot_hull, rot_inner = params
    if r_hull < 0.05 or r_inner < 0.0 or r_inner > r_hull * 0.99:
        return 1e6
    pts = _make_symmetric(n_hull, n_inner, r_hull, r_inner, rot_hull, rot_inner)
    return _soft_min_score(pts, beta)


def _symmetric_hard(params, n_hull, n_inner):
    r_hull, r_inner, rot_hull, rot_inner = params
    if r_hull < 0.05 or r_inner < 0.0 or r_inner > r_hull * 0.99:
        return 1e6
    pts = _make_symmetric(n_hull, n_inner, r_hull, r_inner, rot_hull, rot_inner)
    return -_score(pts)


def _smallest_triangle(pts):
    """Return (min_area, (i, j, k)) for the smallest triangle."""
    areas = _all_areas(pts)
    mi = int(np.argmin(areas))
    # map flat index back to (i,j,k)
    # We precomputed _TRIPLES in the same order as _all_areas
    i, j, k = _TRIPLES[mi]
    return float(areas[mi]), (i, j, k)


def _lift(pts, i, j, k, strength=0.02):
    """Move point i directly away from the line through j and k."""
    pts = pts.copy()
    pj, pk, pi = pts[j], pts[k], pts[i]
    d = pk - pj
    nrm = np.linalg.norm(d)
    if nrm < 1e-12:
        d = pi - pj
        nrm = np.linalg.norm(d)
        if nrm < 1e-12:
            return pts
        n = d / nrm
        perp = np.array([-n[1], n[0]])
        side = np.sign(np.dot(pi - pj, perp))
        if side == 0:
            side = 1.0
        pts[i] = pi + side * perp * strength * nrm
    else:
        n = d / nrm
        perp = np.array([-n[1], n[0]])
        side = np.sign(np.dot(pi - pj, perp))
        if side == 0:
            side = 1.0
        pts[i] = pi + side * perp * strength * nrm
    return pts


def _local_polish(pts, max_iter=4000):
    """Deterministic local search: repeatedly lift the smallest triangle's points."""
    pts = pts.copy()
    cur = _score(pts)
    for _ in range(max_iter):
        if not _time_ok(2.0):
            break
        _, (i, j, k) = _smallest_triangle(pts)
        improved = False
        for pt in (i, j, k):
            if pt == i:
                other_j, other_k = j, k
            elif pt == j:
                other_j, other_k = i, k
            else:
                other_j, other_k = i, j
            for s in (0.003, 0.007, 0.015, 0.03, 0.06, 0.12):
                cand = _lift(pts, pt, other_j, other_k, strength=s)
                sc = _score(cand)
                if sc > cur + 1e-12:
                    pts, cur, improved = cand, sc, True
                    break
            if improved:
                break
        if not improved:
            break
    return pts, cur


def _simulated_anneal(pts, rng, t0=0.015, t1=1e-6, n_steps=25000, lift_frac=0.20):
    """Global SA with random jitter + targeted lift moves."""
    pts = pts.copy()
    cur = _score(pts)
    best = cur
    best_pts = pts.copy()
    n = len(pts)
    for step in range(n_steps):
        if not _time_ok(3.0):
            break
        frac = step / max(1, n_steps)
        T = t0 * (t1 / t0) ** frac
        if rng.rand() < lift_frac:
            _, (i, j, k) = _smallest_triangle(pts)
            pt = int(rng.randint(3))
            if pt == 0:
                cand = _lift(pts, i, j, k, strength=rng.uniform(0.005, 0.05))
            elif pt == 1:
                cand = _lift(pts, j, i, k, strength=rng.uniform(0.005, 0.05))
            else:
                cand = _lift(pts, k, i, j, strength=rng.uniform(0.005, 0.05))
        else:
            cand = pts.copy()
            npts = 1 if rng.rand() < 0.7 else 2
            for _ in range(npts):
                idx = int(rng.randint(n))
                cand[idx] += rng.normal(0, T, size=2)
        sc = _score(cand)
        if sc >= cur or rng.rand() < np.exp((sc - cur) / max(T, 1e-12)):
            cur, pts = sc, cand
            if cur > best:
                best, best_pts = cur, cand.copy()
    return best_pts, best


def construct():
    rng = np.random.RandomState(12345)
    best_score = -1e9
    best_pts = None

    # ---- Phase A: symmetric two-ring optimization (structural manifold) ----
    splits = [(13, 0), (12, 1), (11, 2), (10, 3), (9, 4), (8, 5), (7, 6), (6, 7)]
    for n_hull, n_inner in splits:
        if not _time_ok(15.0):
            break
        r_hull_init = 1.0
        r_inner_init = 0.35 if n_inner > 0 else 0.0
        rot_hull_init = 0.0
        rot_inner_init = (np.pi / n_inner) if n_inner > 1 else 0.0
        x0 = np.array([r_hull_init, r_inner_init, rot_hull_init, rot_inner_init])

        # Multi-beta L-BFGS-B on the smooth surrogate
        for beta in [15.0, 40.0, 100.0]:
            if not _time_ok(12.0):
                break
            try:
                res = opt.minimize(
                    lambda x, b=beta, nh=n_hull, ni=n_inner: _symmetric_objective(x, nh, ni, b),
                    x0,
                    method='L-BFGS-B',
                    bounds=[(0.3, 2.0), (0.0, 0.99), (-np.pi, np.pi), (-np.pi, np.pi)],
                    options={'maxiter': 120, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                x0 = res.x
                pts = _make_symmetric(n_hull, n_inner, *x0)
                s = _score(pts)
                if s > best_score:
                    best_score, best_pts = s, pts.copy()
            except Exception:
                pass

        # Nelder-Mead polish on the hard objective within the manifold
        if not _time_ok(10.0):
            continue
        try:
            res = opt.minimize(
                lambda x, nh=n_hull, ni=n_inner: _symmetric_hard(x, nh, ni),
                x0,
                method='Nelder-Mead',
                options={'maxiter': 250, 'adaptive': True, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            x0 = res.x
            pts = _make_symmetric(n_hull, n_inner, *x0)
            s = _score(pts)
            if s > best_score:
                best_score, best_pts = s, pts.copy()
        except Exception:
            pass

    # ---- Phase B: free-space gradient descent on the soft surrogate ----
    if best_pts is not None and _time_ok(20.0):
        current = best_pts.copy()
        for beta in [25.0, 60.0, 120.0]:
            if not _time_ok(15.0):
                break
            try:
                res = opt.minimize(
                    lambda x, b=beta: _soft_objective(x, b),
                    current.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                optimized = res.x.reshape(-1, 2)
                s = _score(optimized)
                if s > best_score:
                    best_score, best_pts = s, optimized.copy()
                current = optimized
            except Exception:
                pass

    # ---- Phase C: targeted local polish (lift worst triangle) ----
    if best_pts is not None and _time_ok(8.0):
        try:
            lp, ls = _local_polish(best_pts, max_iter=5000)
            if ls > best_score:
                best_score, best_pts = ls, lp.copy()
        except Exception:
            pass

    # ---- Phase D: simulated annealing + polish for global escape ----
    if best_pts is not None and _time_ok(10.0):
        try:
            ap, ascr = _simulated_anneal(best_pts, rng, t0=0.012, t1=1e-7, n_steps=20000, lift_frac=0.25)
            if ascr > best_score:
                best_score, best_pts = ascr, ap.copy()
        except Exception:
            pass
        if _time_ok(5.0):
            try:
                lp, ls = _local_polish(best_pts, max_iter=4000)
                if ls > best_score:
                    best_score, best_pts = ls, lp.copy()
            except Exception:
                pass

    # ---- Phase E: a few random restarts in free space (if time permits) ----
    if _time_ok(12.0):
        for _ in range(5):
            if not _time_ok(6.0):
                break
            # random points in unit disk, biased toward two rings
            n_hull = int(rng.randint(9, 13))
            n_inner = N - n_hull
            r_h = rng.uniform(0.7, 1.2)
            r_i = rng.uniform(0.1, 0.6) if n_inner > 0 else 0.0
            pts = _make_symmetric(n_hull, n_inner, r_h, r_i,
                                  rng.uniform(-np.pi, np.pi),
                                  rng.uniform(-np.pi, np.pi))
            # add small jitter
            pts += rng.normal(0, 0.02, size=pts.shape)
            s = _score(pts)
            if s > best_score:
                best_score, best_pts = s, pts.copy()
            # quick local polish
            if _time_ok(4.0):
                try:
                    lp, ls = _local_polish(pts, max_iter=1500)
                    if ls > best_score:
                        best_score, best_pts = ls, lp.copy()
                except Exception:
                    pass

    # Final safety: if something went wrong, fall back to a regular 13-gon
    if best_pts is None or best_score <= 0:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_pts = np.column_stack([np.cos(angles), np.sin(angles)])

    return best_pts
