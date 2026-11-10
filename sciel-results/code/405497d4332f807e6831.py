import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def _vertex_score(v, pts):
    """Compute the vertex score for vertex v given all points pts."""
    diffs = pts - v
    dists = np.sqrt(np.sum(diffs**2, axis=1))
    mask = np.ones(len(pts), dtype=bool)
    mask[np.argmin(dists)] = False
    d = np.sort(dists[mask])
    if np.any(d < 0.05):
        return 0.0
    gaps = np.diff(d)
    k = int(np.argmin(gaps))
    D = (d[k] + d[k + 1]) / 2.0
    if D < 0.1:
        return 0.0
    if k == 0:
        selected = d[0:4]
    elif k == 17:
        selected = d[15:19]
    else:
        selected = d[k - 1:k + 3]
    ratios = np.maximum(selected / D, D / selected)
    raw = np.mean(ratios)
    return min(1.0, 1.0 / raw)

def _min_vertex_score(pts):
    scores = [_vertex_score(pts[i], pts) for i in range(len(pts))]
    return min(scores)

def _is_valid(pts):
    if len(pts) != 20:
        return False
    if not np.all(np.isfinite(pts)):
        return False
    for i in range(20):
        for j in range(i + 1, 20):
            if np.linalg.norm(pts[i] - pts[j]) < 1e-12:
                return False
    signs = []
    for i in range(20):
        p0 = pts[i]
        p1 = pts[(i + 1) % 20]
        p2 = pts[(i + 2) % 20]
        v1 = p1 - p0
        v2 = p2 - p1
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        if abs(cross) < 1e-14:
            return False
        signs.append(np.sign(cross))
    if not all(s == signs[0] for s in signs):
        return False
    return True

def _normalize(pts):
    centroid = np.mean(pts, axis=0)
    pts_norm = pts - centroid
    dists = np.linalg.norm(pts_norm, axis=1)
    mean_dist = np.mean(dists)
    if mean_dist <= 0.01:
        return pts_norm, False
    pts_norm = pts_norm / mean_dist
    return pts_norm, True

def _polar_to_pts(polar_params):
    """
    Convert polar parameters (angles, radii) to Cartesian points.
    polar_params: shape (40,) -> first 20 are angles, next 20 are radii.
    """
    angles = polar_params[:20]
    radii = polar_params[20:]
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return np.column_stack([x, y])

def _objective(polar_params):
    """Negative min vertex score (we minimize)."""
    pts = _polar_to_pts(polar_params)
    if not np.all(np.isfinite(pts)):
        return 1.0  # bad
    norm_pts, valid = _normalize(pts)
    if not valid:
        return 1.0
    if not _is_valid(norm_pts):
        return 1.0
    score = _min_vertex_score(norm_pts)
    return -score

def _regular_polar():
    """Return polar parameters for regular 20-gon."""
    angles = np.linspace(0, 2 * np.pi, 20, endpoint=False)
    radii = np.ones(20)
    return np.concatenate([angles, radii])

def equidistant_points_polygon20() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 590.0  # safety margin

    n = 20
    # Start with regular polygon in polar coords
    best_polar = _regular_polar()
    best_pts = _polar_to_pts(best_polar)
    norm_best, valid_best = _normalize(best_pts)
    if valid_best and _is_valid(norm_best):
        best_score = _min_vertex_score(norm_best)
    else:
        best_score = 0.0

    # Phase 1: Differential Evolution (global search)
    # Bounds: angles in [0, 2*pi], radii in [0.3, 3.0]
    angle_bounds = [(0.0, 2.0 * np.pi)] * 20
    radius_bounds = [(0.3, 3.0)] * 20
    bounds = angle_bounds + radius_bounds

    elapsed = time.time() - start_time
    if elapsed < deadline - 30:
        try:
            de_result = differential_evolution(
                _objective,
                bounds=bounds,
                seed=42,
                maxiter=50,
                popsize=15,
                tol=1e-4,
                polish=False,
                init='sobol',
                updating='immediate',
                workers=-1
            )
            if de_result.fun < -best_score:
                best_polar = de_result.x
                best_pts = _polar_to_pts(best_polar)
                norm_best, valid_best = _normalize(best_pts)
                if valid_best and _is_valid(norm_best):
                    best_score = _min_vertex_score(norm_best)
        except Exception:
            pass

    elapsed = time.time() - start_time
    if elapsed < deadline - 15 and best_score < 1.0:
        # Phase 2: Nelder-Mead local refinement from DE result
        try:
            nm_result = minimize(
                _objective,
                x0=best_polar,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            if nm_result.fun < -best_score:
                best_polar = nm_result.x
                best_pts = _polar_to_pts(best_polar)
                norm_best, valid_best = _normalize(best_pts)
                if valid_best and _is_valid(norm_best):
                    best_score = _min_vertex_score(norm_best)
        except Exception:
            pass

    elapsed = time.time() - start_time
    if elapsed < deadline - 10 and best_score < 1.0:
        # Phase 3: Additional Nelder-Mead from regular polygon (different basin)
        try:
            regular_polar = _regular_polar()
            nm_result2 = minimize(
                _objective,
                x0=regular_polar,
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True}
            )
            if nm_result2.fun < -best_score:
                best_polar = nm_result2.x
                best_pts = _polar_to_pts(best_polar)
                norm_best, valid_best = _normalize(best_pts)
                if valid_best and _is_valid(norm_best):
                    best_score = _min_vertex_score(norm_best)
        except Exception:
            pass

    elapsed = time.time() - start_time
    if elapsed < deadline - 5 and best_score < 1.0:
        # Phase 4: A few random restarts with Nelder-Mead
        rng = np.random.default_rng(123)
        for restart in range(10):
            if time.time() > deadline - 5:
                break
            # Random perturbation of regular polygon in polar space
            angles = np.linspace(0, 2 * np.pi, 20, endpoint=False) + rng.normal(0, 0.1, 20)
            radii = 1.0 + rng.normal(0, 0.1, 20)
            radii = np.clip(radii, 0.3, 3.0)
            angles = angles % (2 * np.pi)
            polar_init = np.concatenate([angles, radii])
            try:
                nm_r = minimize(
                    _objective,
                    x0=polar_init,
                    method='Nelder-Mead',
                    options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-9, 'adaptive': True}
                )
                if nm_r.fun < -best_score:
                    best_polar = nm_r.x
                    best_pts = _polar_to_pts(best_polar)
                    norm_best, valid_best = _normalize(best_pts)
                    if valid_best and _is_valid(norm_best):
                        best_score = _min_vertex_score(norm_best)
            except Exception:
                continue

    return best_pts
