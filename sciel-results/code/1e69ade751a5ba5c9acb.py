"""Heilbronn triangles - feasibility-driven bisection with SLSQP."""
N = 11
H = np.sqrt(3) / 2 if False else None  # placeholder, set below

import numpy as np
import time
from itertools import combinations
from scipy.optimize import minimize

_H = np.sqrt(3) / 2
_EPS = 1e-10

# Precompute index triples (165 of them)
_IDX = np.array(list(combinations(range(N), 3)))
_I = _IDX[:, 0]
_J = _IDX[:, 1]
_K = _IDX[:, 2]
_NC = len(_IDX)


def _triangle_areas(p):
    """All 165 triangle areas (unsigned), vectorized."""
    pi = p[_I]; pj = p[_J]; pk = p[_K]
    v1x = pj[:, 0] - pi[:, 0]; v1y = pj[:, 1] - pi[:, 1]
    v2x = pk[:, 0] - pi[:, 0]; v2y = pk[:, 1] - pi[:, 1]
    cross = v1x * v2y - v1y * v2x
    return 0.5 * np.abs(cross)


def _min_area(p):
    return float(np.min(_triangle_areas(p)))


def _clamp(p):
    """Clamp all points inside the equilateral triangle."""
    x = p[:, 0].copy(); y = p[:, 1].copy()
    y = np.clip(y, _EPS, _H - _EPS)
    xl = y / np.sqrt(3)
    xr = 1.0 - y / np.sqrt(3)
    x = np.clip(x, xl + _EPS, xr - _EPS)
    return np.column_stack([x, y])


def _slsqp_feasible(p0, alpha, time_limit=30.0):
    """
    Try to find a point set (near p0) where every triangle area >= alpha.
    Returns (success, points).
    """
    t0 = time.time()
    p0 = _clamp(np.array(p0, dtype=float))

    def objective(x):
        p = x.reshape(N, 2)
        areas = _triangle_areas(p)
        return -np.min(areas)

    # Constraints: area_k - alpha >= 0  for each triangle k
    # SLSQP wants g(x) >= 0 for 'ineq'. We provide g = areas - alpha.
    def cons(x):
        p = x.reshape(N, 2)
        return _triangle_areas(p) - alpha

    # Equality constraints: fix the 3 vertices (0,0),(1,0),(0.5,H) to anchor scale/position.
    # This reduces the search space and prevents trivial translation/scale drift.
    # Point indices: 0 -> (0,0), 1 -> (1,0), 2 -> (0.5, H)
    eq = []
    # x0 = 0
    eq.append({'type': 'eq', 'fun': lambda x: x[0], 'jac': lambda x: np.array([1.0, 0.0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])})
    # y0 = 0
    eq.append({'type': 'eq', 'fun': lambda x: x[1], 'jac': lambda x: np.array([0.0, 1.0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])})
    # x1 = 1
    eq.append({'type': 'eq', 'fun': lambda x: x[2], 'jac': lambda x: np.array([0,0,1.0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])})
    # y1 = 0
    eq.append({'type': 'eq', 'fun': lambda x: x[3], 'jac': lambda x: np.array([0,0,0,1.0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])})
    # x2 = 0.5
    eq.append({'type': 'eq', 'fun': lambda x: x[4], 'jac': lambda x: np.array([0,0,0,0,1.0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])})
    # y2 = H
    eq.append({'type': 'eq', 'fun': lambda x: x[5], 'jac': lambda x: np.array([0,0,0,0,0,1.0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])})

    # Bounds: keep all points inside the triangle (loose bounds; clamp handles exact)
    bounds = [(0.0, 1.0), (0.0, _H)] * N

    x0 = p0.flatten()
    try:
        res = minimize(
            objective, x0, method='SLSQP',
            bounds=bounds,
            constraints=[{'type': 'ineq', 'fun': cons}] + eq,
            options={'maxiter': 400, 'ftol': 1e-10, 'disp': False}
        )
    except Exception:
        return False, p0

    if time.time() - t0 > time_limit:
        return False, p0

    p = _clamp(res.x.reshape(N, 2))
    if _min_area(p) >= alpha - 1e-9:
        return True, p
    return False, p


def _slsqp_maximize(p0, time_limit=40.0):
    """Maximize min triangle area directly with SLSQP (smooth surrogate via min of areas)."""
    t0 = time.time()
    p0 = _clamp(np.array(p0, dtype=float))

    def objective(x):
        if time.time() - t0 > time_limit:
            return -1e6
        p = x.reshape(N, 2)
        return -_min_area(p)

    def cons(x):
        p = x.reshape(N, 2)
        return _triangle_areas(p)

    bounds = [(0.0, 1.0), (0.0, _H)] * N
    x0 = p0.flatten()
    try:
        res = minimize(
            objective, x0, method='SLSQP',
            bounds=bounds,
            constraints=[{'type': 'ineq', 'fun': cons}],
            options={'maxiter': 600, 'ftol': 1e-10, 'disp': False}
        )
    except Exception:
        return p0
    return _clamp(res.x.reshape(N, 2))


def _move_critical(p, alpha_target, time_limit=15.0):
    """
    Targeted local search: nudge the points that participate in the
    smallest triangles to increase the minimum area.
    """
    t0 = time.time()
    p = _clamp(np.array(p, dtype=float))
    best = _min_area(p)
    areas = _triangle_areas(p)
    # Critical triangles: those with area near the minimum
    for _ in range(200):
        if time.time() - t0 > time_limit:
            break
        areas = _triangle_areas(p)
        mn = np.min(areas)
        if mn >= best:
            best = mn
        # Pick the worst triangle
        worst = int(np.argmin(areas))
        i, j, k = _IDX[worst]
        # Nudge one of the three points away from the others
        for idx in (i, j, k):
            others = [q for q in (i, j, k) if q != idx]
            # Direction: away from centroid of the other two
            cx = (p[others[0], 0] + p[others[1], 0]) / 2.0
            cy = (p[others[0], 1] + p[others[1], 1]) / 2.0
            dx = p[idx, 0] - cx
            dy = p[idx, 1] - cy
            nrm = np.hypot(dx, dy)
            if nrm < 1e-12:
                dx, dy = np.random.randn(2)
                nrm = np.hypot(dx, dy)
            step = 0.01
            cand = p.copy()
            cand[idx, 0] += step * dx / nrm
            cand[idx, 1] += step * dy / nrm
            cand = _clamp(cand)
            if _min_area(cand) > best:
                p = cand
                best = _min_area(p)
                break
    return p, best


def _initial_configs():
    """Diverse starting points, including near-symmetric layouts."""
    configs = []
    # Symmetric: 3 vertices + concentric
    sym = np.array([
        [0.0, 0.0], [1.0, 0.0], [0.5, _H],
        [0.5, _H/3], [0.25, _H/6], [0.75, _H/6],
        [0.5, _H/2], [0.125, _H/12], [0.875, _H/12],
        [0.375, _H/4], [0.625, _H/4]
    ])
    configs.append(sym)

    # Triangular lattice-ish
    lat = np.array([
        [0.0, 0.0], [0.5, 0.0], [1.0, 0.0],
        [0.25, _H/4], [0.75, _H/4],
        [0.5, _H/2],
        [0.125, _H/8], [0.375, _H/8], [0.625, _H/8], [0.875, _H/8],
        [0.5, _H/6]
    ])
    configs.append(lat)

    # Random seeds
    rng = np.random.default_rng(7)
    for s in range(6):
        pts = rng.random((N, 2))
        pts[:, 1] *= _H
        # Force vertices
        pts[0] = [0.0, 0.0]; pts[1] = [1.0, 0.0]; pts[2] = [0.5, _H]
        configs.append(_clamp(pts))
    return configs


def construct():
    t_start = time.time()
    deadline = t_start + 130  # 20s safety margin before 150s

    configs = _initial_configs()
    best_p = None
    best_score = -1.0

    # Phase 0: evaluate configs, pick a few good seeds
    scored = []
    for c in configs:
        c = _clamp(c)
        scored.append((_min_area(c), c))
    scored.sort(key=lambda t: -t[0])
    top = [c for _, c in scored[:4]]

    # Phase 1: SLSQP maximize from each top seed (bounded time)
    for c in top:
        if time.time() > deadline - 40:
            break
        tl = min(35.0, max(10.0, deadline - 40 - (time.time() - t_start)))
        try:
            p = _slsqp_maximize(c, time_limit=tl)
            s = _min_area(p)
            if s > best_score:
                best_score = s
                best_p = p
        except Exception:
            pass

    # Phase 2: Bisection on alpha using SLSQP feasibility, starting from best
    if best_p is not None and time.time() < deadline - 30:
        lo = best_score
        hi = min(0.05, best_score * 1.5 + 0.005)  # upper guess
        # First, find a feasible upper bound
        for _ in range(8):
            if time.time() > deadline - 30:
                break
            mid = 0.5 * (lo + hi)
            ok, p = _slsqp_feasible(best_p, mid, time_limit=20.0)
            if ok:
                lo = mid
                s = _min_area(p)
                if s > best_score:
                    best_score = s
                    best_p = p
            else:
                hi = mid
        # Refine around lo
        for _ in range(6):
            if time.time() > deadline - 25:
                break
            mid = 0.5 * (lo + hi)
            if mid <= lo:
                break
            ok, p = _slsqp_feasible(best_p, mid, time_limit=15.0)
            if ok:
                lo = mid
                s = _min_area(p)
                if s > best_score:
                    best_score = s
                    best_p = p
            else:
                hi = mid

    # Phase 3: Critical-triangle polish
    if best_p is not None and time.time() < deadline - 5:
        tl = max(3.0, deadline - 5 - (time.time() - t_start))
        try:
            p, s = _move_critical(best_p, best_score, time_limit=tl)
            if s > best_score:
                best_score = s
                best_p = p
        except Exception:
            pass

    # Phase 4: Final SLSQP polish
    if best_p is not None and time.time() < deadline - 3:
        tl = max(2.0, deadline - 3 - (time.time() - t_start))
        try:
            p = _slsqp_maximize(best_p, time_limit=tl)
            s = _min_area(p)
            if s > best_score:
                best_score = s
                best_p = p
        except Exception:
            pass

    if best_p is None:
        best_p = _clamp(configs[0])
    return _clamp(best_p)
