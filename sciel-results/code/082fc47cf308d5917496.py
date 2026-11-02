"""Heilbronn convex regions: 13 points maximizing min triangle area / hull area."""
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull

N = 13
DEADLINE = 140.0  # safety margin before 150s kill


def _min_tri_area(pts):
    """Exact minimum triangle area among all triples."""
    n = len(pts)
    min_area = np.inf
    for i in range(n):
        vi = pts - pts[i]
        for j in range(i + 1, n):
            v = pts[j] - pts[i]
            cross = vi[:, 0] * v[1] - vi[:, 1] * v[0]
            areas = np.abs(cross) * 0.5
            areas[i] = np.inf
            areas[j] = np.inf
            m = np.min(areas)
            if m < min_area:
                min_area = m
    return min_area


def _hull_area(pts):
    try:
        return ConvexHull(pts).volume
    except Exception:
        order = np.argsort(np.arctan2(pts[:, 1] - pts[:, 1].mean(),
                                      pts[:, 0] - pts[:, 0].mean()))
        p = pts[order]
        x, y = p[:, 0], p[:, 1]
        return 0.5 * np.abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _score(pts):
    hull_a = _hull_area(pts)
    if hull_a < 1e-12:
        return 0.0
    return _min_tri_area(pts) / hull_a


def _all_tri_areas(pts):
    """All 286 triangle areas (absolute, 0.5*|cross|)."""
    n = len(pts)
    areas = []
    for i in range(n):
        vi = pts - pts[i]
        for j in range(i + 1, n):
            v = pts[j] - pts[i]
            cross = vi[:, 0] * v[1] - vi[:, 1] * v[0]
            a = np.abs(cross) * 0.5
            a = np.delete(a, j)
            a = np.delete(a, i)
            areas.append(a)
    return np.concatenate(areas)


def _soft_min_score(params, beta):
    """Smooth surrogate: -softmin(tri_areas)/hull_area. Returns negative for minimization."""
    pts = _build(params)
    hull_a = _hull_area(pts)
    if hull_a < 1e-12:
        return 1e6
    areas = _all_tri_areas(pts)
    # soft min via log-sum-exp with sign flip
    m = np.min(areas)
    if m < 1e-12:
        return 1e6
    # softmin = m * (1 - (1/beta) * log(sum(exp(-beta*(a-m))))/K)  ... use stable form
    z = -beta * (areas - m)
    z = z - np.max(z)
    s = np.sum(np.exp(z))
    soft = m - (1.0 / beta) * np.log(s)
    return -(soft / hull_a)


def _build(params):
    """Build 13 points from ring structure: 7 outer + 6 inner, centered at origin."""
    r_out, r_in, ang_out, ang_in = params[0], params[1], params[2], params[3]
    ao = np.linspace(0, 2 * np.pi, 7, endpoint=False) + ang_out
    ai = np.linspace(0, 2 * np.pi, 6, endpoint=False) + ang_in
    p_out = np.column_stack([r_out * np.cos(ao), r_out * np.sin(ao)])
    p_in = np.column_stack([r_in * np.cos(ai), r_in * np.sin(ai)])
    return np.vstack([p_out, p_in])


def _build_alt(params):
    """Alt: 6 outer hexagon + 7 inner heptagon."""
    r_out, r_in, ang_out, ang_in = params[0], params[1], params[2], params[3]
    ao = np.linspace(0, 2 * np.pi, 6, endpoint=False) + ang_out
    ai = np.linspace(0, 2 * np.pi, 7, endpoint=False) + ang_in
    p_out = np.column_stack([r_out * np.cos(ao), r_out * np.sin(ao)])
    p_in = np.column_stack([r_in * np.cos(ai), r_in * np.sin(ai)])
    return np.vstack([p_out, p_in])


def _neg_score_flat(x):
    pts = x.reshape(-1, 2)
    return -_score(pts)


def construct():
    t_start = time.time()
    best_score = 0.0
    best_pts = None

    def try_update(pts):
        nonlocal best_score, best_pts
        if pts is None:
            return
        try:
            s = _score(pts)
        except Exception:
            return
        if s > best_score:
            best_score = s
            best_pts = pts.copy()

    # ---- Phase 1: coarse global search in 4D symmetric space (soft surrogate) ----
    for build_fn, bounds in [
        (_build, [(0.5, 1.5), (0.05, 1.0), (0, 2 * np.pi / 7), (0, 2 * np.pi / 6)]),
        (_build_alt, [(0.5, 1.5), (0.05, 1.0), (0, 2 * np.pi / 6), (0, 2 * np.pi / 7)]),
    ]:
        if time.time() - t_start > DEADLINE - 30:
            break
        try:
            res = differential_evolution(
                _soft_min_score, bounds, args=(20.0,),
                seed=42, maxiter=60, popsize=12,
                tol=1e-6, mutation=(0.4, 1.2), recombination=0.8, polish=False
            )
            try_update(build_fn(res.x))
        except Exception:
            continue

    # ---- Phase 2: refine in 4D with exact objective via Powell ----
    for build_fn, bounds in [
        (_build, [(0.5, 1.5), (0.05, 1.0), (0, 2 * np.pi / 7), (0, 2 * np.pi / 6)]),
        (_build_alt, [(0.5, 1.5), (0.05, 1.0), (0, 2 * np.pi / 6), (0, 2 * np.pi / 7)]),
    ]:
        if time.time() - t_start > DEADLINE - 20:
            break
        def neg_sym(p):
            return -_score(build_fn(p))
        try:
            res = minimize(neg_sym, np.array(bounds).mean(axis=1),
                           method='Powell', options={'maxiter': 400, 'ftol': 1e-12})
            try_update(build_fn(res.x))
        except Exception:
            continue

    # ---- Phase 3: full 26D refinement from best (Nelder-Mead) ----
    if best_pts is not None and time.time() - t_start < DEADLINE - 15:
        try:
            res = minimize(_neg_score_flat, best_pts.flatten(),
                           method='Nelder-Mead',
                           options={'maxiter': 3000, 'xatol': 1e-9, 'fatol': 1e-12})
            try_update(res.x.reshape(-1, 2))
        except Exception:
            pass

    # ---- Phase 4: multi-start Nelder-Mead with diverse seeds ----
    if time.time() - t_start < DEADLINE - 10:
        starts = []
        # regular 13-gon
        a13 = np.linspace(0, 2 * np.pi, 13, endpoint=False)
        starts.append(np.column_stack([np.cos(a13), np.sin(a13)]).flatten())
        # symmetric configs
        for r_out in [0.7, 0.9, 1.1]:
            for r_in in [0.3, 0.5, 0.7]:
                starts.append(_build([r_out, r_in, 0, 0]).flatten())
                starts.append(_build_alt([r_out, r_in, 0, 0]).flatten())
        if best_pts is not None:
            rng = np.random.RandomState(0)
            for _ in range(6):
                starts.append((best_pts + rng.randn(13, 2) * 0.1).flatten())
        for x0 in starts:
            if time.time() - t_start > DEADLINE:
                break
            try:
                res = minimize(_neg_score_flat, x0, method='Nelder-Mead',
                               options={'maxiter': 600, 'xatol': 1e-8, 'fatol': 1e-10})
                try_update(res.x.reshape(-1, 2))
            except Exception:
                continue

    if best_pts is None:
        a13 = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_pts = np.column_stack([np.cos(a13), np.sin(a13)])
    return best_pts


if __name__ == "__main__":
    pts = construct()
    print(f"Score: {_score(pts)}")
    print(f"Points:\n{pts}")
