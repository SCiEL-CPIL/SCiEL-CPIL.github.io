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


def construct():
    t_start = time.time()
    best_score = -float('inf')
    best_params = None

    def _update(pts):
        nonlocal best_score, best_params
        if pts is None or pts.shape != (N, 2):
            return
        s = _score_points(pts)
        if s > best_score:
            best_score = s
            best_params = pts.copy()

    # ============================================================
    # Phase 1: Grid search over symmetric layered configurations
    # ============================================================
    # Strategy A: Two-layer (hull polygon + inner polygon)
    hull_radii = np.linspace(0.5, 1.5, 10)
    inner_radii = np.linspace(0.0, 0.7, 10)
    offset_vals = np.linspace(0, np.pi / 2, 5)

    for n_hull in range(3, N + 1):
        n_inner = N - n_hull
        if n_inner < 0:
            continue
        for hr in hull_radii:
            if time.time() - t_start > DEADLINE - 40:
                break
            if n_inner == 0:
                for ho in offset_vals:
                    cand = _make_two_layer(n_hull, hr, 0.0, hull_offset=ho)
                    _update(cand)
            else:
                for ir in inner_radii:
                    if time.time() - t_start > DEADLINE - 40:
                        break
                    for ho in offset_vals:
                        for io in offset_vals:
                            if time.time() - t_start > DEADLINE - 40:
                                break
                            cand = _make_two_layer(n_hull, hr, ir,
                                                   hull_offset=ho, inner_offset=io)
                            _update(cand)

    # ============================================================
    # Phase 2: Differential evolution (global search)
    # ============================================================
    # Initial guess: regular 13-gon
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

    # Use best grid result as additional seed
    if best_params is not None:
        _update(best_params)

    bounds = [(-1.5, 1.5)] * (2 * N)

    try:
        result = differential_evolution(
            _neg_score,
            bounds,
            seed=42,
            maxiter=25,
            popsize=12,
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
    # Phase 3: Local refinement with L-BFGS-B from multiple starts
    # ============================================================
    starts = []
    if best_params is not None:
        starts.append(best_params.flatten().copy())
    starts.append(x0.flatten().copy())

    # Add perturbed versions
    rng = np.random.RandomState(42)
    if best_params is not None:
        for _ in range(4):
            pert = best_params.flatten() + rng.normal(0, 0.08, size=2 * N)
            starts.append(pert)
        for _ in range(2):
            pert = x0.flatten() + rng.normal(0, 0.15, size=2 * N)
            starts.append(pert)

    for s in starts:
        if time.time() - t_start > DEADLINE - 10:
            break
        try:
            res = minimize(
                _neg_score,
                s,
                method='L-BFGS-B',
                bounds=[(-2.0, 2.0)] * (2 * N),
                options={'maxiter': 400, 'maxfun': 8000, 'ftol': 1e-12}
            )
            cand = res.x.reshape(N, 2)
            _update(cand)
        except Exception:
            pass

    # ============================================================
    # Phase 4: Nelder-Mead polish on best result
    # ============================================================
    if best_params is not None and time.time() - t_start < DEADLINE - 5:
        try:
            res_nm = minimize(
                _neg_score,
                best_params.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 600, 'xatol': 1e-10, 'fatol': 1e-12,
                         'adaptive': True}
            )
            cand = res_nm.x.reshape(N, 2)
            _update(cand)
        except Exception:
            pass

    # ============================================================
    # Phase 5: Physics-inspired repulsion refinement
    # Push points apart using inverse-square repulsion,
    # then re-optimize with L-BFGS-B
    # ============================================================
    if best_params is not None and time.time() - t_start < DEADLINE - 8:
        try:
            pts = best_params.copy()
            # Iterative repulsion: push each point away from others
            for _iter in range(30):
                forces = np.zeros_like(pts)
                for i in range(N):
                    for j in range(N):
                        if i == j:
                            continue
                        d = pts[i] - pts[j]
                        dist_sq = np.dot(d, d)
                        if dist_sq < 1e-12:
                            d = rng.normal(0, 1e-4, size=2)
                            dist_sq = np.dot(d, d)
                        # Inverse-square repulsion
                        f = d / (dist_sq * np.sqrt(dist_sq))
                        forces[i] += f
                # Also add a gentle attraction toward the centroid to prevent divergence
                centroid = np.mean(pts, axis=0)
                for i in range(N):
                    forces[i] -= 0.01 * (pts[i] - centroid)
                pts += 0.005 * forces
            _update(pts)

            # Polish the repulsion result
            if time.time() - t_start < DEADLINE - 3:
                res_rep = minimize(
                    _neg_score,
                    pts.flatten(),
                    method='L-BFGS-B',
                    bounds=[(-2.0, 2.0)] * (2 * N),
                    options={'maxiter': 300, 'maxfun': 6000, 'ftol': 1e-12}
                )
                cand = res_rep.x.reshape(N, 2)
                _update(cand)
        except Exception:
            pass

    # ============================================================
    # Phase 6: Final random restarts if time permits
    # ============================================================
    attempts = 0
    while (time.time() - t_start) < (DEADLINE - 5) and attempts < 6:
        attempts += 1
        if best_params is None:
            break
        pert = best_params.flatten() + rng.normal(0, 0.05, size=2 * N)
        try:
            res = minimize(
                _neg_score,
                pert,
                method='L-BFGS-B',
                bounds=[(-2.0, 2.0)] * (2 * N),
                options={'maxiter': 200, 'maxfun': 4000, 'ftol': 1e-12}
            )
            cand = res.x.reshape(N, 2)
            _update(cand)
        except Exception:
            pass

    if best_params is None:
        return x0
    return best_params
