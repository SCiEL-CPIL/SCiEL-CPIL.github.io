import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 14
# Precompute triangle indices
_INDICES = np.array([(i, j, k) for i in range(N) for j in range(i + 1, N) for k in range(j + 1, N)])
_I_IDX = _INDICES[:, 0]
_J_IDX = _INDICES[:, 1]
_K_IDX = _INDICES[:, 2]
_NTRIS = len(_INDICES)


def _get_all_triangle_areas(points):
    """Areas of all C(N,3) triangles (vectorized)."""
    p_i = points[_I_IDX]
    p_j = points[_J_IDX]
    p_k = points[_K_IDX]
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross_z = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross_z)


def _hull_area(points):
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0


def _normalize(points):
    """Center and scale so convex hull area == 1."""
    p = points - np.mean(points, axis=0)
    ha = _hull_area(p)
    if ha < 1e-10:
        return p
    return p / np.sqrt(ha)


def _score(points):
    n = len(points)
    if n < 3:
        return 0.0
    p = _normalize(points)
    ha = _hull_area(p)
    if ha < 1e-10:
        return 0.0
    areas = _get_all_triangle_areas(p)
    return float(np.min(areas) / ha)


def _softmin_neg(points_flat, tau):
    """Negative of the soft-min of triangle areas (to maximize min area).
    Returns f = -softmin(areas) where softmin ≈ min.
    """
    points = points_flat.reshape(-1, 2)
    points = points - np.mean(points, axis=0)
    ha = _hull_area(points)
    if ha < 1e-10:
        return 1e6
    points = points / np.sqrt(ha)
    areas = _get_all_triangle_areas(points)
    # soft min = -1/tau * log(sum exp(-tau * areas))
    a_min = np.min(areas)
    shifted = areas - a_min
    # numerical stability
    sum_exp = np.sum(np.exp(-tau * shifted))
    softmin = a_min - (1.0 / tau) * np.log(sum_exp)
    return -softmin


def _exact_neg(points_flat):
    """Negative of exact min triangle area (normalized)."""
    points = points_flat.reshape(-1, 2)
    points = points - np.mean(points, axis=0)
    ha = _hull_area(points)
    if ha < 1e-10:
        return 1e6
    points = points / np.sqrt(ha)
    areas = _get_all_triangle_areas(points)
    return -np.min(areas)


def _make_regular_14gon():
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    pts = np.column_stack([np.cos(angles), np.sin(angles)])
    return _normalize(pts)


def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s kill

    best_points = _make_regular_14gon()
    best_score = _score(best_points)

    def elapsed():
        return time.time() - start_time

    # Phase 1: Soft-min annealing from the regular polygon.
    # Start with moderate tau (smoother, more global), decrease toward exact min.
    taus = [8.0, 20.0, 50.0, 150.0, 400.0, 1200.0, 4000.0]
    x = best_points.flatten().copy()

    for tau in taus:
        if elapsed() > deadline - 25.0:
            break
        try:
            res = minimize(
                _softmin_neg, x, args=(tau,), method='Powell',
                options={'maxiter': 600, 'xtol': 1e-10, 'ftol': 1e-12,
                         'adaptive': True, 'niter': 80}
            )
            x = res.x
            p = _normalize(x.reshape(-1, 2))
            s = _score(p)
            if s > best_score:
                best_score = s
                best_points = p
                x = best_points.flatten().copy()
        except Exception:
            pass

    # Phase 2: Exact refinement with Nelder-Mead (non-smooth min).
    if elapsed() < deadline - 20.0:
        try:
            res = minimize(
                _exact_neg, best_points.flatten(), method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-13, 'fatol': 1e-15,
                         'adaptive': True, 'maxfev': 6000}
            )
            p = _normalize(res.x.reshape(-1, 2))
            s = _score(p)
            if s > best_score:
                best_score = s
                best_points = p
        except Exception:
            pass

    # Phase 3: Targeted refinement — nudge points that participate in the
    # smallest triangles to increase the minimum.
    if elapsed() < deadline - 10.0:
        p = best_points.copy()
        for it in range(60):
            if elapsed() > deadline - 5.0:
                break
            areas = _get_all_triangle_areas(p)
            m = np.min(areas)
            # indices of triangles near the minimum
            near = np.where(areas <= m + 1e-4 * m + 1e-8)[0]
            # collect the points involved
            involved = set()
            for t in near:
                involved.add(int(_I_IDX[t]))
                involved.add(int(_J_IDX[t]))
                involved.add(int(_K_IDX[t]))
            # small random perturbation of involved points, keep others fixed
            rng = np.random.default_rng(int(elapsed() * 1000) + it)
            p_new = p.copy()
            for idx in involved:
                p_new[idx] += rng.normal(0, 0.01, size=2)
            p_new = _normalize(p_new)
            s = _score(p_new)
            if s > best_score + 1e-12:
                best_score = s
                best_points = p_new
                p = p_new
            # also try a few larger steps occasionally
            if it % 5 == 0:
                p_new2 = p.copy()
                for idx in involved:
                    p_new2[idx] += rng.normal(0, 0.05, size=2)
                p_new2 = _normalize(p_new2)
                s2 = _score(p_new2)
                if s2 > best_score + 1e-12:
                    best_score = s2
                    best_points = p_new2
                    p = p_new2

    # Phase 4: Final polish with Nelder-Mead from the best point.
    if elapsed() < deadline - 3.0:
        try:
            res = minimize(
                _exact_neg, best_points.flatten(), method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-14, 'fatol': 1e-16,
                         'adaptive': True, 'maxfev': 4000}
            )
            p = _normalize(res.x.reshape(-1, 2))
            s = _score(p)
            if s > best_score:
                best_score = s
                best_points = p
        except Exception:
            pass

    # Ensure final normalization
    best_points = _normalize(best_points)
    return best_points
