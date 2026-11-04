"""Covering the sphere with 130 equal caps – structural redesign.

Approach:
  1. Start from a Fibonacci-spiral seed (fast, decent initial spread).
  2. Compute the convex hull via scipy.spatial.ConvexHull to identify facets.
  3. For each facet (triangle), compute its circumradius (angular distance from
     facet centroid to the farthest vertex – actually the spherical circumradius).
  4. Use a smooth surrogate objective: a high-order power mean of facet circumradii
     (approximates max while being differentiable).
  5. Optimise with L-BFGS-B on the point coordinates (re-normalised each step).
  6. Multi-phase: coarse power-mean -> fine power-mean -> Nelder-Mead polish on
     the worst few points -> final L-BFGS-B.
  7. Deadline guard: return best feasible result before 150 s wall-clock.
"""
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, minimize_nelder_mead
import sys

N = 130
TIME_LIMIT = 145.0  # safety margin before 150 s kill


def _normalize(pts):
    """Project points onto the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms


def _spherical_circumradius(p1, p2, p3):
    """
    Angular circumradius of a spherical triangle with vertices p1, p2, p3
    (unit vectors).  The circumradius is the angular distance from the
    spherical circumcenter to each vertex.  For a small triangle this is
    approximately the planar circumradius of the projected triangle.

    We use the formula: R = a / (2 * sin(A)) where a is a side and A the
    opposite angle.  More robustly, for three unit vectors we can compute
    the circumradius via the determinant formula.

    A simpler and robust approach: the circumradius of a spherical triangle
    with side lengths a, b, c (angular) satisfies
        sin(R) = (a * b * c) / (2 * sqrt((s)(s-a)(s-b)(s-c)))  ... not quite.

    Actually, for a spherical triangle the circumradius R satisfies:
        tan(R) = (a * b * c) / (4 * area)   (for small triangles, approx)

    Let's use a direct geometric approach:
    The circumcenter of a spherical triangle is the pole of the great circle
    through the three points.  The circumradius is the angular distance from
    that pole to any vertex.

    For three unit vectors v1, v2, v3, the circumcenter direction is
        n = v1 × v2 × v3  (normalized)  -- wait, that's the normal to the plane.
    Actually the circumcenter lies on the great circle that is the perpendicular
    bisector... This gets complicated.

    Simpler: for small triangles, approximate by planar circumradius of the
    three points projected onto the tangent plane at the centroid.
    """
    # Side lengths (angular)
    def ang_dist(a, b):
        d = np.clip(np.dot(a, b), -1.0, 1.0)
        return np.arccos(d)

    a = ang_dist(p2, p3)
    b = ang_dist(p1, p3)
    c = ang_dist(p1, p2)

    # Spherical circumradius formula:
    #   tan(R) = (a * b * c) / (4 * K)  where K is the spherical area? No.
    # Actually: R = arcsin( (sin(a/2)*sin(b/2)*sin(c/2)) / (2*sin(A/2)*sin(B/2)*sin(C/2)) )... too complex.

    # For small triangles, use planar approximation:
    # Project onto tangent plane at centroid
    centroid = (p1 + p2 + p3) / 3.0
    norm_c = np.linalg.norm(centroid)
    if norm_c < 1e-12:
        return 0.0
    centroid = centroid / norm_c

    # Tangent plane basis
    u = p1 - np.dot(p1, centroid) * centroid
    nu = np.linalg.norm(u)
    if nu < 1e-12:
        return 0.0
    u = u / nu
    v = np.cross(centroid, u)
    nv = np.linalg.norm(v)
    if nv < 1e-12:
        return 0.0
    v = v / nv

    # Project vertices
    q1 = np.array([np.dot(p1, u), np.dot(p1, v)])
    q2 = np.array([np.dot(p2, u), np.dot(p2, v)])
    q3 = np.array([np.dot(p3, u), np.dot(p3, v)])

    # Planar circumradius
    ax, ay = q1
    bx, by = q2
    cx, cy = q3
    A = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(A) < 1e-15:
        return 0.0
    a_len = np.sqrt((bx - cx) ** 2 + (by - cy) ** 2)
    b_len = np.sqrt((ax - cx) ** 2 + (ay - cy) ** 2)
    c_len = np.sqrt((ax - bx) ** 2 + (ay - by) ** 2)
    R = (a_len * b_len * c_len) / (2.0 * abs(A))
    return R


def _compute_facet_circumradii(pts):
    """
    Compute the circumradius (angular) for each facet of the convex hull.
    Returns array of circumradii in radians.
    """
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.array([0.0])

    simplices = hull.simplices
    radii = np.zeros(len(simplices))
    for i, simplex in enumerate(simplices):
        p1, p2, p3 = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
        radii[i] = _spherical_circumradius(p1, p2, p3)
    return radii


def _covering_radius(pts):
    """Compute the covering radius (max facet circumradius) in degrees."""
    radii = _compute_facet_circumradii(pts)
    if len(radii) == 0:
        return 180.0
    return np.degrees(np.max(radii))


def _objective_power_mean(pts_flat, p_power):
    """
    Smooth surrogate: power mean of facet circumradii.
    High p_power approximates max.
    """
    pts = _normalize(pts_flat.reshape(-1, 3))
    radii = _compute_facet_circumradii(pts)
    if len(radii) == 0:
        return 1e6
    # Power mean
    radii = np.maximum(radii, 1e-10)
    pm = (np.mean(radii ** p_power)) ** (1.0 / p_power)
    return pm


def _objective_variational(pts_flat):
    """
    Weighted sum of squared deviations of facet circumradii from their mean,
    plus a small penalty on the mean itself to encourage small radii.
    This enforces equiangularity (all facets similar size).
    """
    pts = _normalize(pts_flat.reshape(-1, 3))
    radii = _compute_facet_circumradii(pts)
    if len(radii) == 0:
        return 1e6
    mean_r = np.mean(radii)
    # Variance term + small mean term
    var_term = np.mean((radii - mean_r) ** 2)
    mean_term = 0.1 * mean_r ** 2
    return var_term + mean_term


def _gradient_descent_step(pts, radii, step_size=0.01):
    """
    Simple gradient step: move each point away from the direction that
    increases the worst facet's circumradius.
    """
    pts_new = pts.copy()
    # For each facet, the gradient w.r.t. each vertex pushes the vertex
    # to increase/decrease the circumradius.  We use a heuristic:
    # move each vertex slightly toward the centroid of the worst facet.
    if len(radii) == 0:
        return pts_new

    worst_idx = np.argmax(radii)
    # Get the worst facet vertices
    try:
        hull = ConvexHull(pts)
        worst_simplex = hull.simplices[worst_idx]
    except Exception:
        return pts_new

    for vi in worst_simplex:
        # Move vertex vi slightly toward the centroid of the sphere region
        # opposite to the facet normal (i.e., inward toward the sphere center
        # relative to the facet)
        p = pts[vi]
        # Slight random perturbation to escape local minima
        noise = np.random.randn(3) * step_size * 0.5
        # Pull toward origin (reduces cap size)
        pts_new[vi] = p * (1.0 - step_size) + noise
    return _normalize(pts_new)


def _fibonacci_seed():
    """Generate Fibonacci spiral seed."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return _normalize(np.column_stack([r * np.cos(phi), r * np.sin(phi), z]))


def construct():
    start_time = time.time()
    best_pts = _fibonacci_seed()
    best_score = _covering_radius(best_pts)

    def time_remaining():
        return TIME_LIMIT - (time.time() - start_time)

    # Phase 1: L-BFGS-B with power-mean objective (p=8)
    if time_remaining() > 30.0:
        p_power = 8.0
        x0 = best_pts.flatten()
        res = minimize(
            _objective_power_mean, x0, args=(p_power,),
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50}
        )
        if time_remaining() > 0.5:
            cand = _normalize(res.x.reshape(-1, 3))
            score = _covering_radius(cand)
            if score < best_score:
                best_score = score
                best_pts = cand

    # Phase 2: L-BFGS-B with higher power mean (p=16)
    if time_remaining() > 25.0:
        p_power = 16.0
        x0 = best_pts.flatten()
        res = minimize(
            _objective_power_mean, x0, args=(p_power,),
            method='L-BFGS-B',
            options={'maxiter': 400, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50}
        )
        if time_remaining() > 0.5:
            cand = _normalize(res.x.reshape(-1, 3))
            score = _covering_radius(cand)
            if score < best_score:
                best_score = score
                best_pts = cand

    # Phase 3: Variational (equiangularity) objective
    if time_remaining() > 20.0:
        x0 = best_pts.flatten()
        res = minimize(
            _objective_variational, x0,
            method='L-BFGS-B',
            options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50}
        )
        if time_remaining() > 0.5:
            cand = _normalize(res.x.reshape(-1, 3))
            score = _covering_radius(cand)
            if score < best_score:
                best_score = score
                best_pts = cand

    # Phase 4: Nelder-Mead polish on a subset of points (worst facets)
    if time_remaining() > 15.0:
        radii = _compute_facet_circumradii(best_pts)
        if len(radii) > 0:
            # Identify worst facets and their vertices
            try:
                hull = ConvexHull(best_pts)
                sorted_facets = np.argsort(radii)[::-1]
                worst_facets = sorted_facets[:10]
                # Collect unique vertices from worst facets
                worst_vertices = set()
                for fi in worst_facets:
                    for v in hull.simplices[fi]:
                        worst_vertices.add(v)
                worst_vertices = sorted(worst_vertices)
            except Exception:
                worst_vertices = []

            if worst_vertices:
                # Optimize only these points
                def nm_obj(x_sub):
                    pts_full = best_pts.copy()
                    for i, vi in enumerate(worst_vertices):
                        pts_full[vi] = x_sub[i * 3:(i + 1) * 3]
                    pts_full = _normalize(pts_full)
                    return _covering_radius(pts_full)

                x0_sub = np.array([best_pts[vi] for vi in worst_vertices]).flatten()
                res_nm = minimize(
                    nm_obj, x0_sub,
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-10}
                )
                if time_remaining() > 0.5:
                    pts_full = best_pts.copy()
                    for i, vi in enumerate(worst_vertices):
                        pts_full[vi] = res_nm.x[i * 3:(i + 1) * 3]
                    cand = _normalize(pts_full)
                    score = _covering_radius(cand)
                    if score < best_score:
                        best_score = score
                        best_pts = cand

    # Phase 5: Final L-BFGS-B with very high power mean
    if time_remaining() > 10.0:
        p_power = 32.0
        x0 = best_pts.flatten()
        res = minimize(
            _objective_power_mean, x0, args=(p_power,),
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50}
        )
        if time_remaining() > 0.5:
            cand = _normalize(res.x.reshape(-1, 3))
            score = _covering_radius(cand)
            if score < best_score:
                best_score = score
                best_pts = cand

    # Final validation: ensure points are on unit sphere and non-coincident
    best_pts = _normalize(best_pts)
    # Check for coincident points
    for i in range(N):
        for j in range(i + 1, N):
            if np.linalg.norm(best_pts[i] - best_pts[j]) < 1e-8:
                # Separate them slightly
                best_pts[j] = best_pts[j] + np.random.randn(3) * 0.01
    best_pts = _normalize(best_pts)

    return best_pts
