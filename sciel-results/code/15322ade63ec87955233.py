"""
Heilbronn convex regions n=14: maximize min triangle area / convex hull area.

Strategy:
  1. Centrally symmetric parameterization (exploits known structure for even N=14).
  2. Multi-phase optimization: global DE in symmetric space + local NM/Powell/L-BFGS.
  3. Fast vectorized triangle-area computation with precomputed indices.
  4. Time-guarded to finish well within 150s.
"""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull
import itertools

N = 14
M = N // 2  # 7 points in half-space, the other 7 are their negatives

# Pre-compute all triangle index combinations for full 14 points
tri_indices = np.array(list(itertools.combinations(range(N), 3)))  # shape (364, 3)


def _min_triangle_area(points):
    """Minimum triangle area over all triples, vectorized."""
    p = points
    pi = p[tri_indices[:, 0]]
    pj = p[tri_indices[:, 1]]
    pk = p[tri_indices[:, 2]]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    return np.min(areas)


def _all_triangle_areas(points):
    pi = points[tri_indices[:, 0]]
    pj = points[tri_indices[:, 1]]
    pk = points[tri_indices[:, 2]]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    return 0.5 * np.abs(cross)


def _hull_area(points):
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0


def _score(points):
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_area = _min_triangle_area(points)
    return min_area / hull_area


def _objective_hard(flat):
    pts = flat.reshape(N, 2)
    if np.any(~np.isfinite(pts)):
        return 1e10
    min_area = _min_triangle_area(pts)
    if min_area < 1e-12:
        return 1e10
    hull_area = _hull_area(pts)
    if hull_area < 1e-10:
        return 1e10
    return hull_area / min_area


def _objective_soft(flat, beta=60.0):
    pts = flat.reshape(N, 2)
    if np.any(~np.isfinite(pts)):
        return 1e10
    areas = _all_triangle_areas(pts)
    if np.min(areas) < 1e-12:
        return 1e10
    z = -beta * areas
    max_z = np.max(z)
    if max_z < -500:
        log_sum = max_z
    else:
        log_sum = max_z + np.log(np.sum(np.exp(z - max_z)))
    soft_min = -log_sum / beta
    if soft_min < 1e-12:
        return 1e10
    hull_area = _hull_area(pts)
    if hull_area < 1e-10:
        return 1e10
    return hull_area / soft_min


# ---------- Centrally symmetric parameterization ----------
# For centrally symmetric configurations, point i and point i+7 are negatives.
# We parameterize the first 7 points (14 dims), the rest are determined.

def _sym_params_to_points(sym_params):
    """Convert 14-dim symmetric params to 14x2 points."""
    half = sym_params.reshape(M, 2)
    return np.vstack([half, -half])


def _points_to_sym_params(points):
    """Extract first 7 points from 14-point centrally symmetric config."""
    return points[:M].flatten()


def _objective_sym(sym_params):
    pts = _sym_params_to_points(sym_params)
    return _objective_hard(pts.flatten())


def _objective_sym_soft(sym_params, beta=60.0):
    pts = _sym_params_to_points(sym_params)
    return _objective_soft(pts.flatten(), beta)


# ---------- Optimizer helpers ----------

def _nm(x0, deadline, obj, max_iter=2000):
    if time.time() > deadline - 3:
        return None
    try:
        res = minimize(obj, x0, method='Nelder-Mead',
                       options={'maxiter': max_iter, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True})
        return res
    except Exception:
        return None


def _powell(x0, deadline, obj, max_iter=1000):
    if time.time() > deadline - 3:
        return None
    try:
        res = minimize(obj, x0, method='Powell',
                       options={'maxiter': max_iter, 'xtol': 1e-11, 'ftol': 1e-13})
        return res
    except Exception:
        return None


def _lbfgs(x0, deadline, obj, max_iter=3000):
    if time.time() > deadline - 5:
        return None
    try:
        res = minimize(obj, x0, method='L-BFGS-B',
                       options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10})
        return res
    except Exception:
        return None


def _normalize_unit_area(points):
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area > 1e-10:
            scale = np.sqrt(1.0 / hull_area)
            return points * scale
    except Exception:
        pass
    xs = points[:, 0]
    ys = points[:, 1]
    w = xs.max() - xs.min()
    h = ys.max() - ys.min()
    if w > 1e-10 and h > 1e-10:
        scale = 1.0 / np.sqrt(w * h)
        return points * scale
    return points


# ---------- Initial configurations (symmetric params: 14-dim) ----------

def _generate_sym_inits():
    """Generate diverse initial symmetric parameter vectors (14-dim)."""
    inits = []
    rng = np.random.default_rng(42)

    # 1. Regular 7-gon (half of regular 14-gon)
    angles = np.linspace(0, 2 * np.pi, M, endpoint=False)
    inits.append(np.column_stack([np.cos(angles), np.sin(angles)]).flatten())

    # 2. Regular 7-gon at various radii
    for r in [0.5, 0.7, 0.85, 1.0, 1.2]:
        angles = np.linspace(0, 2 * np.pi, M, endpoint=False)
        inits.append(r * np.column_stack([np.cos(angles), np.sin(angles)]).flatten())

    # 3. Rotated regular 7-gons
    for r in [0.7, 1.0]:
        for rot in [np.pi / 14, np.pi / 7, 2 * np.pi / 7]:
            angles = np.linspace(0, 2 * np.pi, M, endpoint=False) + rot
            inits.append(r * np.column_stack([np.cos(angles), np.sin(angles)]).flatten())

    # 4. Elliptical 7-gons
    for rx, ry in [(1.0, 0.6), (1.0, 0.8), (1.2, 0.7), (0.8, 1.0)]:
        a = np.linspace(0, 2 * np.pi, M, endpoint=False)
        inits.append(np.column_stack([rx * np.cos(a), ry * np.sin(a)]).flatten())

    # 5. Jittered regular 7-gon
    base = np.column_stack([np.cos(angles), np.sin(angles)])
    for js in [0.05, 0.1, 0.15, 0.2]:
        inits.append((base + rng.normal(0, js, size=(M, 2))).flatten())

    # 6. Random symmetric configs
    for seed in range(15):
        rs = np.random.default_rng(seed * 13 + 5)
        angles = np.sort(rs.uniform(0, 2 * np.pi, M))
        radii = rs.uniform(0.4, 1.3, M)
        inits.append(np.column_stack([radii * np.cos(angles), radii * np.sin(angles)]).flatten())

    # 7. Two-ring in symmetric space (3+4 split)
    for r1, r2 in [(0.5, 1.0), (0.6, 1.0), (0.7, 1.1)]:
        a1 = np.linspace(0, 2 * np.pi, 3, endpoint=False)
        a2 = np.linspace(0, 2 * np.pi, 4, endpoint=False) + np.pi / 4
        pts = np.vstack([
            r1 * np.column_stack([np.cos(a1), np.sin(a1)]),
            r2 * np.column_stack([np.cos(a2), np.sin(a2)])
        ])
        inits.append(pts.flatten())

    return inits


# ---------- Main ----------

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill

    best_score = 0.0
    best_pts = None

    def _consider(pts):
        nonlocal best_score, best_pts
        pts = np.asarray(pts, dtype=float)
        if pts.shape != (N, 2) or np.any(~np.isfinite(pts)):
            return
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts.copy()

    # Phase 1: Differential evolution in symmetric space (14 dims)
    if time.time() < deadline - 60:
        try:
            bounds = [(-1.5, 1.5)] * (M * 2)
            res_de = differential_evolution(
                _objective_sym, bounds,
                seed=42, maxiter=50, tol=1e-10,
                popsize=20, mutation=(0.4, 1.0), recombination=0.7,
                polish=False, update_interval=1
            )
            if res_de is not None and np.isfinite(res_de.fun) and res_de.fun < 1e9:
                pts = _sym_params_to_points(res_de.x)
                _consider(pts)
        except Exception:
            pass

    # Phase 2: NM refinement from diverse symmetric inits
    sym_inits = _generate_sym_inits()
    if best_pts is not None:
        sym_inits.insert(0, _points_to_sym_params(best_pts))

    for i, params0 in enumerate(sym_inits):
        if time.time() > deadline - 45:
            break
        res = _nm(params0, deadline, _objective_sym, max_iter=1500)
        if res is not None and np.isfinite(res.fun) and res.fun < 1e9:
            pts = _sym_params_to_points(res.x)
            _consider(pts)

    # Phase 3: Powell refinement from best in symmetric space
    if best_pts is not None and time.time() < deadline - 35:
        sym_params = _points_to_sym_params(best_pts)
        res = _powell(sym_params, deadline, _objective_sym, max_iter=800)
        if res is not None and np.isfinite(res.fun) and res.fun < 1e9:
            pts = _sym_params_to_points(res.x)
            _consider(pts)

    # Phase 4: L-BFGS on soft objective in symmetric space
    if best_pts is not None and time.time() < deadline - 25:
        sym_params = _points_to_sym_params(best_pts)
        res = _lbfgs(sym_params, deadline, _objective_sym_soft, max_iter=2000)
        if res is not None and np.isfinite(res.fun) and res.fun < 1e9:
            pts = _sym_params_to_points(res.x)
            _consider(pts)

    # Phase 5: Second DE round with tighter bounds around best
    if best_pts is not None and time.time() < deadline - 20:
        try:
            sym_best = _points_to_sym_params(best_pts)
            bounds2 = [(sym_best[i] - 0.3, sym_best[i] + 0.3) for i in range(M * 2)]
            res_de2 = differential_evolution(
                _objective_sym, bounds2,
                seed=123, maxiter=30, tol=1e-10,
                popsize=15, mutation=(0.3, 0.8), recombination=0.6,
                polish=False, update_interval=1
            )
            if res_de2 is not None and np.isfinite(res_de2.fun) and res_de2.fun < 1e9:
                pts = _sym_params_to_points(res_de2.x)
                _consider(pts)
        except Exception:
            pass

    # Phase 6: NM local refinement after second DE
    if best_pts is not None and time.time() < deadline - 12:
        sym_params = _points_to_sym_params(best_pts)
        res = _nm(sym_params, deadline, _objective_sym, max_iter=2000)
        if res is not None and np.isfinite(res.fun) and res.fun < 1e9:
            pts = _sym_params_to_points(res.x)
            _consider(pts)

    # Phase 7: Powell final polish in symmetric space
    if best_pts is not None and time.time() < deadline - 8:
        sym_params = _points_to_sym_params(best_pts)
        res = _powell(sym_params, deadline, _objective_sym, max_iter=600)
        if res is not None and np.isfinite(res.fun) and res.fun < 1e9:
            pts = _sym_params_to_points(res.x)
            _consider(pts)

    # Phase 8: Final L-BFGS polish in symmetric space
    if best_pts is not None and time.time() < deadline - 5:
        sym_params = _points_to_sym_params(best_pts)
        res = _lbfgs(sym_params, deadline, _objective_sym_soft, max_iter=1500)
        if res is not None and np.isfinite(res.fun) and res.fun < 1e9:
            pts = _sym_params_to_points(res.x)
            _consider(pts)

    # Fallback
    if best_pts is None:
        angles = np.linspace(0, 2 * np.pi, M, endpoint=False)
        half = np.column_stack([np.cos(angles), np.sin(angles)])
        best_pts = np.vstack([half, -half])

    final_points = _normalize_unit_area(best_pts)

    # Safety check
    try:
        hull = ConvexHull(final_points)
        if hull.volume < 1e-10:
            angles = np.linspace(0, 2 * np.pi, M, endpoint=False)
            half = np.column_stack([np.cos(angles), np.sin(angles)])
            final_points = _normalize_unit_area(np.vstack([half, -half]))
    except Exception:
        angles = np.linspace(0, 2 * np.pi, M, endpoint=False)
        half = np.column_stack([np.cos(angles), np.sin(angles)])
        final_points = _normalize_unit_area(np.vstack([half, -half]))

    return final_points
