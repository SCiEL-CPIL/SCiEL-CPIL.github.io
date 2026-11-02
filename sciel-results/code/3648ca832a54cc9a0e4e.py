"""
Heilbronn convex regions n=14: maximize min triangle area / convex hull area.

Strategy:
  1. Use a radial (Fourier) parameterization for fast global exploration.
  2. Use a log-barrier constrained Cartesian optimization for local refinement
     that can break radial symmetry.
  3. Combine both in a multi-phase pipeline with time guards.
"""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull
from itertools import combinations

N = 14

# Pre-compute all triangle index combinations
tri_indices = np.array(list(combinations(range(N), 3)))  # shape (364, 3)

# Fixed angles for radial parameterization
angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
sin_delta = np.sin(2 * np.pi / N)

# Number of harmonics for radial expansion
M_HARMONICS = 6


def _min_triangle_area(points):
    """Minimum triangle area over all triples (vectorized)."""
    n = len(points)
    if n < 3:
        return 0.0
    pi = points[tri_indices[:, 0]]
    pj = points[tri_indices[:, 1]]
    pk = points[tri_indices[:, 2]]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return np.min(0.5 * np.abs(cross))


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


# ---------- Radial (Fourier) parameterization ----------

def _get_radii_from_coeffs(coeffs):
    """
    coeffs: array of length 2*M containing [a1, b1, a2, b2, ..., aM, bM]
    Returns: array of N radii (positive).
    """
    r = np.ones(N)
    for m in range(1, M_HARMONICS + 1):
        a = coeffs[2 * (m - 1)]
        b = coeffs[2 * (m - 1) + 1]
        r += a * np.cos(m * angles) + b * np.sin(m * angles)
    r = np.maximum(r, 0.05)
    return r


def _compute_score_from_radii(r):
    """
    Compute min triangle area / convex hull area given radii.
    Uses the fast polygon area formula for convex star-shaped polygons.
    """
    x = r * np.cos(angles)
    y = r * np.sin(angles)
    pts = np.column_stack([x, y])

    # Convex hull area via shoelace (valid for convex polygon in angular order)
    r_next = np.roll(r, -1)
    hull_area = 0.5 * np.sum(r * r_next) * sin_delta
    if hull_area < 1e-10:
        return 0.0

    # Vectorized min triangle area
    pi = pts[tri_indices[:, 0]]
    pj = pts[tri_indices[:, 1]]
    pk = pts[tri_indices[:, 2]]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    min_area = np.min(areas)

    return min_area / hull_area


def _neg_score_coeffs(coeffs):
    r = _get_radii_from_coeffs(coeffs)
    score = _compute_score_from_radii(r)
    return -score


def _neg_score_radii(r_flat):
    r = np.maximum(r_flat, 0.05)
    score = _compute_score_from_radii(r)
    return -score


# ---------- Log-barrier Cartesian optimization ----------

def _convexity_barrier(flat, mu=0.01):
    """
    Compute a log-barrier penalty for violating convexity of the point set.
    We check that consecutive points (in the given order) form a convex polygon.
    For a convex polygon, all cross products of consecutive edges must be positive.
    We use the log of the minimum cross product as a barrier.
    """
    pts = flat.reshape(N, 2)
    # Compute cross products of consecutive edges
    # Edge i: pts[i+1] - pts[i] (cyclic)
    edges = np.roll(pts, -1, axis=0) - pts
    # Cross product of edge i and edge i+1
    cross_products = edges[:, 0] * np.roll(edges, -1, axis=0)[:, 1] - \
                     edges[:, 1] * np.roll(edges, -1, axis=0)[:, 0]
    # For convexity, all cross products should be positive (same orientation)
    # Use the minimum to determine the barrier
    min_cross = np.min(cross_products)
    if min_cross < 1e-8:
        return 1e10
    # Log barrier: -mu * log(min_cross)
    return -mu * np.log(min_cross)


def _objective_cartesian_barrier(flat, mu=0.01):
    """
    Objective: hull_area / min_triangle_area + convexity barrier.
    """
    pts = flat.reshape(N, 2)
    if np.any(~np.isfinite(pts)):
        return 1e10
    min_area = _min_triangle_area(pts)
    if min_area < 1e-12:
        return 1e10
    hull_area = _hull_area(pts)
    if hull_area < 1e-10:
        return 1e10
    barrier = _convexity_barrier(flat, mu)
    return hull_area / min_area + barrier


def _objective_cartesian_soft(flat, beta=50.0, mu=0.01):
    """
    Soft objective using log-sum-exp for min area + convexity barrier.
    """
    pts = flat.reshape(N, 2)
    if np.any(~np.isfinite(pts)):
        return 1e10

    pi = pts[tri_indices[:, 0]]
    pj = pts[tri_indices[:, 1]]
    pk = pts[tri_indices[:, 2]]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)

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

    barrier = _convexity_barrier(flat, mu)
    return hull_area / soft_min + barrier


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


def _make_initial_configs():
    """Generate diverse initial point configurations."""
    inits = []
    rng = np.random.default_rng(42)

    # 1. Regular 14-gon
    inits.append(np.column_stack([np.cos(angles), np.sin(angles)]))

    # 2. Two concentric rings (7+7)
    for r1 in [0.3, 0.5, 0.7, 0.85]:
        a1 = np.linspace(0, 2 * np.pi, 7, endpoint=False)
        a2 = np.linspace(0, 2 * np.pi, 7, endpoint=False) + np.pi / 7
        pts = np.vstack([
            r1 * np.column_stack([np.cos(a1), np.sin(a1)]),
            np.column_stack([np.cos(a2), np.sin(a2)])
        ])
        inits.append(pts)

    # 3. Three rings (5+4+5)
    for r1, r2 in [(0.3, 0.6), (0.4, 0.75), (0.5, 0.85)]:
        a1 = np.linspace(0, 2 * np.pi, 5, endpoint=False)
        a2 = np.linspace(0, 2 * np.pi, 4, endpoint=False) + np.pi / 4
        a3 = np.linspace(0, 2 * np.pi, 5, endpoint=False) + np.pi / 5
        pts = np.vstack([
            r1 * np.column_stack([np.cos(a1), np.sin(a1)]),
            r2 * np.column_stack([np.cos(a2), np.sin(a2)]),
            np.column_stack([np.cos(a3), np.sin(a3)])
        ])
        inits.append(pts)

    # 4. Random configurations
    for seed in range(10):
        rs = np.random.default_rng(seed * 17 + 3)
        pts = rs.uniform(-1, 1, size=(N, 2))
        inits.append(pts)

    # 5. Jittered regular polygon
    base = np.column_stack([np.cos(angles), np.sin(angles)])
    for js in [0.05, 0.1, 0.2]:
        inits.append(base + rng.normal(0, js, size=(N, 2)))

    # 6. Elliptical configurations
    for rx, ry in [(1.0, 0.6), (1.0, 0.8), (1.2, 0.7)]:
        a = np.linspace(0, 2 * np.pi, N, endpoint=False)
        inits.append(np.column_stack([rx * np.cos(a), ry * np.sin(a)]))

    return inits


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill

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

    # ---------- Phase 1: Fourier radial optimization ----------
    # Optimize the 2*M Fourier coefficients
    x0_coeffs = np.zeros(2 * M_HARMONICS)
    init_score = _compute_score_from_radii(np.ones(N))
    best_score = init_score
    best_r = np.ones(N).copy()

    if time.time() < deadline - 80:
        try:
            res1 = minimize(_neg_score_coeffs, x0_coeffs, method='Powell',
                            options={'maxiter': 1500, 'ftol': 1e-10, 'disp': False})
            if np.isfinite(res1.fun) and -res1.fun > best_score:
                best_r = _get_radii_from_coeffs(res1.x)
                pts = np.column_stack([best_r * np.cos(angles), best_r * np.sin(angles)])
                _consider(pts)
        except Exception:
            pass

    # ---------- Phase 2: Refine individual radii ----------
    if time.time() < deadline - 60:
        try:
            r_start = best_r.copy()
            res2 = minimize(_neg_score_radii, r_start, method='Powell',
                            options={'maxiter': 800, 'ftol': 1e-12, 'disp': False})
            if np.isfinite(res2.fun) and -res2.fun > best_score:
                r_candidate = np.maximum(res2.x, 0.05)
                pts = np.column_stack([r_candidate * np.cos(angles), r_candidate * np.sin(angles)])
                _consider(pts)
                if _score(pts) > best_score:
                    best_r = r_candidate
        except Exception:
            pass

    # ---------- Phase 3: Nelder-Mead on radii ----------
    if time.time() < deadline - 45:
        try:
            res3 = minimize(_neg_score_radii, best_r, method='Nelder-Mead',
                            options={'maxiter': 600, 'xatol': 1e-8, 'fatol': 1e-8, 'disp': False})
            if np.isfinite(res3.fun) and -res3.fun > best_score:
                r_candidate = np.maximum(res3.x, 0.05)
                pts = np.column_stack([r_candidate * np.cos(angles), r_candidate * np.sin(angles)])
                _consider(pts)
        except Exception:
            pass

    # ---------- Phase 4: Cartesian optimization from best radial solution ----------
    if best_pts is not None and time.time() < deadline - 35:
        # Use the best points found so far as starting point
        flat0 = best_pts.flatten()
        try:
            res4 = minimize(_objective_cartesian_soft, flat0, method='L-BFGS-B',
                            options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-10})
            if np.isfinite(res4.fun) and res4.fun < 1e9:
                _consider(res4.x.reshape(N, 2))
        except Exception:
            pass

    # ---------- Phase 5: Nelder-Mead on Cartesian from multiple starts ----------
    if time.time() < deadline - 25:
        inits = _make_initial_configs()
        for i, pts0 in enumerate(inits[:8]):  # Limit to 8 starts for time
            if time.time() > deadline - 15:
                break
            try:
                res5 = minimize(_objective_cartesian_soft, pts0.flatten(),
                                method='Nelder-Mead',
                                options={'maxiter': 800, 'xatol': 1e-9, 'fatol': 1e-11,
                                         'adaptive': True})
                if np.isfinite(res5.fun) and res5.fun < 1e9:
                    _consider(res5.x.reshape(N, 2))
            except Exception:
                pass

    # ---------- Phase 6: Powell refinement on Cartesian ----------
    if best_pts is not None and time.time() < deadline - 15:
        try:
            res6 = minimize(_objective_cartesian_soft, best_pts.flatten(),
                            method='Powell',
                            options={'maxiter': 500, 'xtol': 1e-11, 'ftol': 1e-13})
            if np.isfinite(res6.fun) and res6.fun < 1e9:
                _consider(res6.x.reshape(N, 2))
        except Exception:
            pass

    # ---------- Phase 7: Final L-BFGS polish ----------
    if best_pts is not None and time.time() < deadline - 8:
        try:
            res7 = minimize(_objective_cartesian_soft, best_pts.flatten(),
                            method='L-BFGS-B',
                            options={'maxiter': 1500, 'ftol': 1e-14, 'gtol': 1e-10})
            if np.isfinite(res7.fun) and res7.fun < 1e9:
                _consider(res7.x.reshape(N, 2))
        except Exception:
            pass

    # ---------- Phase 8: Final NM polish ----------
    if best_pts is not None and time.time() < deadline - 5:
        try:
            res8 = minimize(_objective_hard, best_pts.flatten(),
                            method='Nelder-Mead',
                            options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12,
                                     'adaptive': True})
            if np.isfinite(res8.fun) and res8.fun < 1e9:
                _consider(res8.x.reshape(N, 2))
        except Exception:
            pass

    # Fallback
    if best_pts is None:
        best_pts = np.column_stack([np.cos(angles), np.sin(angles)])

    final_points = _normalize_unit_area(best_pts)

    # Safety check
    try:
        hull = ConvexHull(final_points)
        if hull.volume < 1e-10:
            final_points = _normalize_unit_area(np.column_stack([np.cos(angles), np.sin(angles)]))
    except Exception:
        final_points = _normalize_unit_area(np.column_stack([np.cos(angles), np.sin(angles)]))

    return final_points
