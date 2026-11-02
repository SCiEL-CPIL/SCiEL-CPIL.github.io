"""Heilbronn convex regions: maximize min triangle area / convex hull area for 13 points."""
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution, minimize

N = 13
DEADLINE = 138.0  # tighter safety margin for more reliable returns

# Pre-compute all triples for fast vectorized scoring
_triple_i = []
_triple_j = []
_triple_k = []
for _i in range(N):
    for _j in range(_i + 1, N):
        for _k in range(_j + 1, N):
            _triple_i.append(_i)
            _triple_j.append(_j)
            _triple_k.append(_k)
_triple_i = np.array(_triple_i, dtype=np.int32)
_triple_j = np.array(_triple_j, dtype=np.int32)
_triple_k = np.array(_triple_k, dtype=np.int32)


def _score_points(pts):
    """Compute min triangle area / convex hull area for given points."""
    n = len(pts)
    if n < 3:
        return 0.0
    pts = np.asarray(pts, dtype=np.float64)

    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
        if hull_area <= 1e-12:
            return 0.0
    except Exception:
        return 0.0

    x = pts[:, 0]
    y = pts[:, 1]
    xi = x[_triple_i]
    xj = x[_triple_j]
    xk = x[_triple_k]
    yi = y[_triple_i]
    yj = y[_triple_j]
    yk = y[_triple_k]

    cross = np.abs((xj - xi) * (yk - yi) - (yj - yi) * (xk - xi))
    areas = 0.5 * cross
    min_area = np.min(areas)

    if min_area < 1e-15:
        return 0.0

    return min_area / hull_area


def _neg_score(params):
    """Negative score for minimization."""
    pts = params.reshape(N, 2)
    return -_score_points(pts)


def _regular_polygon(n, radius, center=(0.0, 0.0), angle_offset=0.0):
    if n <= 0:
        return np.array([]).reshape(0, 2)
    angles = np.linspace(angle_offset, 2 * np.pi + angle_offset, n, endpoint=False)
    xs = center[0] + radius * np.cos(angles)
    ys = center[1] + radius * np.sin(angles)
    return np.column_stack([xs, ys])


def _make_two_layer(n_hull, hull_radius, inner_radius, hull_offset=0.0, inner_offset=0.0):
    """Two-layer: n_hull polygon vertices + (N-n_hull) inner polygon vertices."""
    n_inner = N - n_hull
    if n_hull < 3 or n_inner < 0:
        return None
    if n_inner == 0:
        return _regular_polygon(n_hull, hull_radius, angle_offset=hull_offset)
    hull_pts = _regular_polygon(n_hull, hull_radius, angle_offset=hull_offset)
    inner_pts = _regular_polygon(n_inner, inner_radius, angle_offset=inner_offset)
    return np.vstack([hull_pts, inner_pts])


def _make_three_layer(n_outer, outer_radius, n_mid, mid_radius, n_inner, inner_radius,
                      outer_offset=0.0, mid_offset=0.0, inner_offset=0.0):
    """Three-layer configuration."""
    if n_outer < 1 or n_mid < 0 or n_inner < 0 or n_outer + n_mid + n_inner != N:
        return None
    pts = []
    if n_outer > 0:
        pts.append(_regular_polygon(n_outer, outer_radius, angle_offset=outer_offset))
    if n_mid > 0:
        pts.append(_regular_polygon(n_mid, mid_radius, angle_offset=mid_offset))
    if n_inner > 0:
        pts.append(_regular_polygon(n_inner, inner_radius, angle_offset=inner_offset))
    return np.vstack(pts)


def _physics_repulsion(pts, n_iter=50, alpha=0.003, center_pull=0.005, rng=None):
    """
    Physics-based repulsion: push points apart using inverse-square forces,
    with gentle center attraction to prevent divergence.
    """
    if rng is None:
        rng = np.random.RandomState(0)
    pts = pts.copy()
    for _iter in range(n_iter):
        forces = np.zeros_like(pts)
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                d = pts[i] - pts[j]
                dist_sq = np.dot(d, d)
                if dist_sq < 1e-14:
                    d = rng.normal(0, 1e-4, size=2)
                    dist_sq = np.dot(d, d)
                # Inverse-square repulsion
                f = d / (dist_sq * np.sqrt(dist_sq))
                forces[i] += f
        # Gentle attraction toward centroid
        centroid = np.mean(pts, axis=0)
        for i in range(N):
            forces[i] -= center_pull * (pts[i] - centroid)
        pts += alpha * forces
    return pts


def construct():
    t_start = time.time()
    best_score = -float('inf')
    best_params = None
    rng = np.random.RandomState(42)

    def _update(pts):
        nonlocal best_score, best_params
        if pts is None or pts.shape != (N, 2):
            return
        s = _score_points(pts)
        if s > best_score:
            best_score = s
            best_params = pts.copy()

    def _time_ok(margin=10):
        return (time.time() - t_start) < (DEADLINE - margin)

    # ============================================================
    # Phase 1: Symmetric ring initializations (key insight from findings)
    # These start in the basin containing score~49.8 solutions
    # ============================================================
    # Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    x0 = np.column_stack([np.cos(angles), np.sin(angles)])
    try:
        hull0 = ConvexHull(x0)
        area0 = hull0.volume
        if area0 > 1e-12:
            x0 = x0 * np.sqrt(1.0 / area0)
    except Exception:
        pass
    _update(x0)

    # 1 center + 12 ring
    for r in np.linspace(0.3, 1.5, 8):
        for offset in np.linspace(0, np.pi / 6, 4):
            ring = _regular_polygon(12, r, angle_offset=offset)
            cand = np.vstack([np.array([[0.0, 0.0]]), ring])
            _update(cand)

    # 1 center + 6 mid + 6 outer
    for r_outer in np.linspace(0.8, 1.8, 6):
        for r_mid in np.linspace(0.3, 0.8, 5):
            for off_o in np.linspace(0, np.pi / 6, 3):
                for off_m in np.linspace(0, np.pi / 6, 3):
                    if not _time_ok(40):
                        break
                    cand = _make_three_layer(6, r_outer, 6, r_mid, 1, 0.0,
                                             outer_offset=off_o, mid_offset=off_m)
                    if cand is not None:
                        _update(cand)

    # Two-layer grid search
    hull_radii = np.linspace(0.5, 1.5, 8)
    inner_radii = np.linspace(0.0, 0.7, 8)
    offset_vals = np.linspace(0, np.pi / 2, 4)

    for n_hull in range(3, N + 1):
        n_inner = N - n_hull
        if n_inner < 0:
            continue
        for hr in hull_radii:
            if not _time_ok(40):
                break
            if n_inner == 0:
                for ho in offset_vals:
                    cand = _make_two_layer(n_hull, hr, 0.0, hull_offset=ho)
                    _update(cand)
            else:
                for ir in inner_radii:
                    if not _time_ok(40):
                        break
                    for ho in offset_vals:
                        for io in offset_vals:
                            if not _time_ok(40):
                                break
                            cand = _make_two_layer(n_hull, hr, ir,
                                                   hull_offset=ho, inner_offset=io)
                            _update(cand)

    # ============================================================
    # Phase 2: Physics-based repulsion from best symmetric config
    # ============================================================
    if best_params is not None and _time_ok(20):
        pts_rep = _physics_repulsion(best_params, n_iter=40, alpha=0.003,
                                     center_pull=0.005, rng=rng)
        _update(pts_rep)

    # ============================================================
    # Phase 3: Differential evolution (global search)
    # ============================================================
    bounds = [(-1.5, 1.5)] * (2 * N)

    try:
        result = differential_evolution(
            _neg_score,
            bounds,
            seed=42,
            maxiter=20,
            popsize=10,
            tol=1e-5,
            mutation=(0.5, 1.0),
            recombination=0.7,
            polish=False,
            init='sobol',
            updating='deferred',
            workers=1
        )
        if result.fun < -best_score:
            candidate = result.x.reshape(N, 2)
            _update(candidate)
    except Exception:
        pass

    # ============================================================
    # Phase 4: Local refinement with L-BFGS-B from multiple starts
    # ============================================================
    starts = []
    if best_params is not None:
        starts.append(best_params.flatten().copy())
    starts.append(x0.flatten().copy())

    # Add perturbed versions
    if best_params is not None:
        for _ in range(5):
            pert = best_params.flatten() + rng.normal(0, 0.06, size=2 * N)
            starts.append(pert)
        for _ in range(3):
            pert = x0.flatten() + rng.normal(0, 0.12, size=2 * N)
            starts.append(pert)

    for s in starts:
        if not _time_ok(10):
            break
        try:
            res = minimize(
                _neg_score,
                s,
                method='L-BFGS-B',
                bounds=[(-2.0, 2.0)] * (2 * N),
                options={'maxiter': 500, 'maxfun': 10000, 'ftol': 1e-13}
            )
            cand = res.x.reshape(N, 2)
            _update(cand)
        except Exception:
            pass

    # ============================================================
    # Phase 5: Nelder-Mead polish on best result
    # ============================================================
    if best_params is not None and _time_ok(8):
        try:
            res_nm = minimize(
                _neg_score,
                best_params.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 800, 'xatol': 1e-10, 'fatol': 1e-13,
                         'adaptive': True}
            )
            cand = res_nm.x.reshape(N, 2)
            _update(cand)
        except Exception:
            pass

    # ============================================================
    # Phase 6: Physics repulsion + L-BFGS polish (combined)
    # ============================================================
    if best_params is not None and _time_ok(10):
        try:
            pts_rep = _physics_repulsion(best_params, n_iter=30, alpha=0.002,
                                         center_pull=0.004, rng=rng)
            _update(pts_rep)

            if _time_ok(5):
                res_rep = minimize(
                    _neg_score,
                    pts_rep.flatten(),
                    method='L-BFGS-B',
                    bounds=[(-2.0, 2.0)] * (2 * N),
                    options={'maxiter': 400, 'maxfun': 8000, 'ftol': 1e-13}
                )
                cand = res_rep.x.reshape(N, 2)
                _update(cand)
        except Exception:
            pass

    # ============================================================
    # Phase 7: Final random restarts if time permits
    # ============================================================
    attempts = 0
    while _time_ok(5) and attempts < 8:
        attempts += 1
        if best_params is None:
            break
        pert = best_params.flatten() + rng.normal(0, 0.04, size=2 * N)
        try:
            res = minimize(
                _neg_score,
                pert,
                method='L-BFGS-B',
                bounds=[(-2.0, 2.0)] * (2 * N),
                options={'maxiter': 300, 'maxfun': 6000, 'ftol': 1e-13}
            )
            cand = res.x.reshape(N, 2)
            _update(cand)
        except Exception:
            pass

    if best_params is None:
        return x0
    return best_params
