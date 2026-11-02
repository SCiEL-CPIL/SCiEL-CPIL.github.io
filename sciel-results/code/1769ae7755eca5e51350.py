"""Heilbronn convex regions: maximize min triangle area / convex hull area for 13 points."""
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution, minimize

N = 13
DEADLINE = 135.0  # safety margin before 150s kill

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
_n_triples = len(_triple_i)


def _hull_area(pts):
    """Compute convex hull area, return 0.0 on failure."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0


def _score_points(pts):
    """Compute min triangle area / convex hull area for given points."""
    n = len(pts)
    if n < 3:
        return 0.0
    pts = np.asarray(pts, dtype=np.float64)

    hull_area = _hull_area(pts)
    if hull_area <= 1e-12:
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


def _smooth_min(areas, k=50.0):
    """Smooth approximation of minimum using log-sum-exp."""
    # soft min: -1/k * log(sum(exp(-k*a_i)))
    neg_k = -k
    max_val = np.max(areas)
    shifted = neg_k * (areas - max_val)
    return max_val + np.log(np.sum(np.exp(shifted))) / k


def _neg_smooth_score(params, k=50.0):
    """Negative smooth-min score for gradient-based optimization."""
    pts = params.reshape(N, 2)
    hull_area = _hull_area(pts)
    if hull_area <= 1e-12:
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
    soft_min = _smooth_min(areas, k)
    return -soft_min / hull_area


def _neg_hard_score(params):
    """Negative exact score for minimization."""
    pts = params.reshape(N, 2)
    return -_score_points(pts)


def _regular_polygon(n, radius, center=(0.0, 0.0), angle_offset=0.0):
    if n <= 0:
        return np.array([]).reshape(0, 2)
    angles = np.linspace(angle_offset, 2 * np.pi + angle_offset, n, endpoint=False)
    xs = center[0] + radius * np.cos(angles)
    ys = center[1] + radius * np.sin(angles)
    return np.column_stack([xs, ys])


def _make_config(n_hull, hull_radius, inner_radius, hull_offset=0.0, inner_offset=0.0):
    """Two-layer configuration: hull polygon + inner polygon."""
    n_inner = N - n_hull
    if n_hull < 3 or n_inner < 0:
        return None
    if n_inner == 0:
        return _regular_polygon(n_hull, hull_radius, angle_offset=hull_offset)
    hull_pts = _regular_polygon(n_hull, hull_radius, angle_offset=hull_offset)
    inner_pts = _regular_polygon(n_inner, inner_radius, angle_offset=inner_offset)
    return np.vstack([hull_pts, inner_pts])


def _normalize_area(pts, target_area=1.0):
    """Scale points so convex hull has target area."""
    hull_area = _hull_area(pts)
    if hull_area <= 1e-12:
        return pts
    scale = np.sqrt(target_area / hull_area)
    return pts * scale


def construct():
    t_start = time.time()
    best_score = -float('inf')
    best_params = None
    rng = np.random.RandomState(2024)

    def _elapsed():
        return time.time() - t_start

    def _update(pts):
        nonlocal best_score, best_params
        if pts is None or pts.shape != (N, 2):
            return
        s = _score_points(pts)
        if s > best_score:
            best_score = s
            best_params = pts.copy()

    # ============================================================
    # Phase 1: Exhaustive symmetric layered search
    # ============================================================
    hull_radii = np.linspace(0.4, 1.8, 15)
    inner_radii = np.linspace(0.0, 0.8, 15)
    offset_vals = np.linspace(0, np.pi, 8, endpoint=False)

    for n_hull in range(3, N + 1):
        n_inner = N - n_hull
        if n_inner < 0:
            continue
        for hr in hull_radii:
            if _elapsed() > DEADLINE - 50:
                break
            if n_inner == 0:
                for ho in offset_vals:
                    cand = _make_config(n_hull, hr, 0.0, hull_offset=ho)
                    if cand is not None:
                        _update(cand)
            else:
                for ir in inner_radii:
                    if _elapsed() > DEADLINE - 50:
                        break
                    for ho in offset_vals:
                        for io in offset_vals:
                            if _elapsed() > DEADLINE - 50:
                                break
                            cand = _make_config(n_hull, hr, ir, hull_offset=ho, inner_offset=io)
                            if cand is not None:
                                _update(cand)

    # ============================================================
    # Phase 2: Differential evolution (global search)
    # ============================================================
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    x0 = np.column_stack([np.cos(angles), np.sin(angles)])
    _update(x0)

    if best_params is not None:
        _update(best_params)

    bounds = [(-1.5, 1.5)] * (2 * N)

    try:
        result = differential_evolution(
            _neg_hard_score,
            bounds,
            seed=123,
            maxiter=30,
            popsize=15,
            tol=1e-5,
            mutation=(0.4, 1.0),
            recombination=0.7,
            polish=False,
            init='sobol',
            updating='immediate',
            workers=1
        )
        if result.fun < -best_score:
            candidate = result.x.reshape(N, 2)
            _update(candidate)
    except Exception:
        pass

    # ============================================================
    # Phase 3: Smooth-min L-BFGS-B refinement from multiple starts
    # ============================================================
    starts = []
    if best_params is not None:
        starts.append(best_params.flatten().copy())
    starts.append(x0.flatten().copy())

    # Add diverse perturbed versions
    if best_params is not None:
        for sigma in [0.05, 0.1, 0.15, 0.2, 0.3]:
            pert = best_params.flatten() + rng.normal(0, sigma, size=2 * N)
            starts.append(pert)

    # Random starts
    for _ in range(5):
        random_pts = rng.uniform(-1, 1, size=(N, 2))
        starts.append(random_pts.flatten().copy())

    for s in starts:
        if _elapsed() > DEADLINE - 15:
            break
        try:
            # Smooth-min optimization with increasing sharpness
            for k in [20.0, 50.0, 100.0]:
                if _elapsed() > DEADLINE - 15:
                    break
                res = minimize(
                    lambda p, k=k: _neg_smooth_score(p, k),
                    s,
                    method='L-BFGS-B',
                    bounds=[(-2.0, 2.0)] * (2 * N),
                    options={'maxiter': 300, 'maxfun': 6000, 'ftol': 1e-12}
                )
                cand = res.x.reshape(N, 2)
                _update(cand)
                s = res.x  # chain: use previous result as next start
        except Exception:
            pass

    # ============================================================
    # Phase 4: Hard-min L-BFGS-B refinement
    # ============================================================
    if best_params is not None:
        starts_hard = [best_params.flatten().copy()]
        for sigma in [0.03, 0.06, 0.1]:
            starts_hard.append(best_params.flatten() + rng.normal(0, sigma, size=2 * N))

        for s in starts_hard:
            if _elapsed() > DEADLINE - 10:
                break
            try:
                res = minimize(
                    _neg_hard_score,
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
    # Phase 5: Nelder-Mead fine polish
    # ============================================================
    if best_params is not None and _elapsed() < DEADLINE - 5:
        try:
            res_nm = minimize(
                _neg_hard_score,
                best_params.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-11, 'fatol': 1e-14, 'adaptive': True}
            )
            cand = res_nm.x.reshape(N, 2)
            _update(cand)
        except Exception:
            pass

    # ============================================================
    # Phase 6: Local search - perturb and re-optimize
    # ============================================================
    attempts = 0
    while _elapsed() < DEADLINE - 5 and attempts < 8:
        attempts += 1
        if best_params is None:
            break
        sigma = 0.04 / (attempts + 1)
        pert = best_params.flatten() + rng.normal(0, sigma, size=2 * N)
        try:
            res = minimize(
                _neg_hard_score,
                pert,
                method='L-BFGS-B',
                bounds=[(-2.0, 2.0)] * (2 * N),
                options={'maxiter': 300, 'maxfun': 6000, 'ftol': 1e-13}
            )
            cand = res.x.reshape(N, 2)
            _update(cand)
        except Exception:
            pass

    # ============================================================
    # Phase 7: Final smooth-min sweep with very high k
    # ============================================================
    if best_params is not None and _elapsed() < DEADLINE - 4:
        try:
            s = best_params.flatten().copy()
            for k in [200.0, 500.0, 1000.0]:
                if _elapsed() > DEADLINE - 4:
                    break
                res = minimize(
                    lambda p, k=k: _neg_smooth_score(p, k),
                    s,
                    method='L-BFGS-B',
                    bounds=[(-2.0, 2.0)] * (2 * N),
                    options={'maxiter': 400, 'maxfun': 8000, 'ftol': 1e-14}
                )
                cand = res.x.reshape(N, 2)
                _update(cand)
                s = res.x
        except Exception:
            pass

    if best_params is None:
        return x0
    return best_params
