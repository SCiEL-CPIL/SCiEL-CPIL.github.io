"""Improved program for covering the sphere with 50 equal caps."""
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 50


def _normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v
    return v / n


def _normalize_points(points):
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    return points / norms


def _get_circumradius(p1, p2, p3):
    """
    Calculate the circumradius of a spherical triangle defined by 3 unit vectors.
    The circumcenter c satisfies c.p1 = c.p2 = c.p3 = cos(R).
    c is parallel to (p1-p2) x (p2-p3).
    """
    n = np.cross(p1 - p2, p2 - p3)
    c = _normalize(n)
    if np.dot(c, p1) < 0:
        c = -c
    cos_r = np.dot(c, p1)
    cos_r = np.clip(cos_r, -1.0, 1.0)
    r = np.arccos(cos_r)
    return r, c


def _compute_covering_radius(points):
    """
    Compute the covering radius of the configuration.
    This is the maximum circumradius of the convex hull facets.
    Returns (max_radius, worst_facet_indices, worst_circumcenter).
    """
    try:
        hull = ConvexHull(points)
        max_r = 0.0
        worst_facet = None
        worst_center = None

        for simplex in hull.simplices:
            p1 = points[simplex[0]]
            p2 = points[simplex[1]]
            p3 = points[simplex[2]]
            r, c = _get_circumradius(p1, p2, p3)
            if r > max_r:
                max_r = r
                worst_facet = simplex
                worst_center = c
        return max_r, worst_facet, worst_center
    except Exception:
        return np.pi, None, None


def _fibonacci_initialization():
    """Generate points using Fibonacci spiral."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _icosahedron_based_initialization():
    """Generate points based on icosahedron vertices and face centers."""
    t = (1.0 + np.sqrt(5.0)) / 2.0
    # 12 icosahedron vertices
    verts = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], dtype=float)
    verts = _normalize_points(verts)

    # 20 face centers (normalized sums of vertices)
    faces = [
        [0, 1, 4], [0, 4, 2], [0, 2, 11], [0, 11, 5], [0, 5, 1],
        [1, 5, 9], [1, 9, 4], [2, 4, 10], [2, 10, 11], [2, 11, 7],
        [3, 7, 10], [3, 10, 6], [3, 6, 8], [3, 8, 7], [4, 9, 8],
        [5, 11, 10], [6, 10, 7], [7, 8, 6], [8, 9, 6], [9, 5, 6]
    ]
    centers = np.array([_normalize(verts[f[0]] + verts[f[1]] + verts[f[2]]) for f in faces])

    # We have 12 + 20 = 32 points. Need 50.
    # Add edge midpoints for some edges to get to 50
    # 30 edges in icosahedron
    edges = [
        (0, 1), (0, 4), (0, 2), (0, 11), (0, 5),
        (1, 5), (1, 9), (1, 4),
        (2, 4), (2, 10), (2, 11), (2, 7),
        (3, 7), (3, 10), (3, 6), (3, 8),
        (4, 9), (4, 10),
        (5, 11), (5, 9), (5, 6),
        (6, 10), (6, 7), (6, 8),
        (7, 8), (7, 10),
        (8, 9), (8, 10),
        (9, 4), (9, 5)
    ]

    # Pick 18 edge midpoints to reach 50 total
    edge_midpoints = []
    for e in edges[:18]:
        mid = _normalize(verts[e[0]] + verts[e[1]])
        edge_midpoints.append(mid)

    all_points = np.vstack([verts, centers, np.array(edge_midpoints)])
    return all_points[:N]


def _jittered_fibonacci_initialization(seed_val=1.0):
    """Fibonacci spiral with deterministic jitter based on seed."""
    pts = _fibonacci_initialization()
    # Apply small deterministic perturbations
    for i in range(N):
        # Deterministic "random" direction based on index
        angle = seed_val * (i + 1) * 0.7
        perturbation = np.array([np.cos(angle), np.sin(angle), np.cos(2 * angle)])
        perturbation = _normalize(perturbation)
        pts[i] = _normalize(pts[i] + 0.05 * perturbation)
    return pts


def _geometric_relaxation(points, max_iters, deadline, step_start=0.05, step_end=0.005):
    """
    Iterative geometric relaxation: move vertices of worst facet towards circumcenter.
    Returns (best_points, best_radius).
    """
    best_points = points.copy()
    best_radius, _, _ = _compute_covering_radius(points)

    for it in range(max_iters):
        if time.time() > deadline:
            break

        current_radius, worst_facet, worst_center = _compute_covering_radius(points)

        if current_radius < best_radius:
            best_radius = current_radius
            best_points = points.copy()

        if worst_facet is None:
            break

        # Adaptive step size
        progress = it / max(max_iters - 1, 1)
        step = step_start * (step_end / step_start) ** progress

        v0 = points[worst_facet[0]]
        v1 = points[worst_facet[1]]
        v2 = points[worst_facet[2]]

        def move_vertex(v, c, step):
            grad = c - np.dot(c, v) * v
            gn = np.linalg.norm(grad)
            if gn < 1e-10:
                return v
            grad = grad / gn
            v_new = v + step * grad
            return _normalize(v_new)

        points[worst_facet[0]] = move_vertex(v0, worst_center, step)
        points[worst_facet[1]] = move_vertex(v1, worst_center, step)
        points[worst_facet[2]] = move_vertex(v2, worst_center, step)

    return best_points, best_radius


def _smooth_surrogate(points, beta=50.0):
    """
    Compute a smooth approximation of the covering radius using log-sum-exp.
    """
    try:
        hull = ConvexHull(points)
        radii = []
        for simplex in hull.simplices:
            p1 = points[simplex[0]]
            p2 = points[simplex[1]]
            p3 = points[simplex[2]]
            r, _ = _get_circumradius(p1, p2, p3)
            radii.append(r)
        radii = np.array(radii)
        if len(radii) == 0:
            return np.pi
        # Log-sum-exp for smooth max
        max_r = np.max(radii)
        if max_r < 1e-10:
            return 0.0
        shifted = radii - max_r
        log_sum = max_r + np.log(np.sum(np.exp(beta * shifted))) / beta
        return log_sum
    except Exception:
        return np.pi


def _smooth_gradient_optimize(points, deadline, beta_start=15.0, beta_end=300.0, max_stages=4):
    """
    Optimize using smooth surrogate with L-BFGS-B.
    Uses a cascade of increasing beta values.
    """
    best_points = points.copy()
    best_radius, _, _ = _compute_covering_radius(points)

    for stage in range(max_stages):
        if time.time() > deadline:
            break

        # Interpolate beta
        progress = stage / max(max_stages - 1, 1)
        beta = beta_start * (beta_end / beta_start) ** progress

        def objective(x):
            pts = _normalize_points(x.reshape(N, 3))
            return _smooth_surrogate(pts, beta)

        def gradient(x):
            pts = _normalize_points(x.reshape(N, 3))
            # Numerical gradient
            eps = 1e-7
            grad = np.zeros_like(x)
            for i in range(N * 3):
                x_plus = x.copy()
                x_plus[i] += eps
                x_minus = x.copy()
                x_minus[i] -= eps
                grad[i] = (objective(x_plus) - objective(x_minus)) / (2 * eps)
            return grad

        x0 = best_points.flatten()
        try:
            result = minimize(
                objective, x0,
                method='L-BFGS-B',
                jac=gradient,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if result.success and result.fun < _smooth_surrogate(best_points, beta):
                candidate = _normalize_points(result.x.reshape(N, 3))
                r, _, _ = _compute_covering_radius(candidate)
                if r < best_radius:
                    best_radius = r
                    best_points = candidate.copy()
        except Exception:
            pass

    return best_points, best_radius


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 20s margin

    # Generate multiple initial configurations
    initializations = [
        _fibonacci_initialization(),
        _icosahedron_based_initialization(),
        _jittered_fibonacci_initialization(1.0),
        _jittered_fibonacci_initialization(2.5),
        _jittered_fibonacci_initialization(4.7),
        _jittered_fibonacci_initialization(7.3),
        _jittered_fibonacci_initialization(11.1),
    ]

    best_points = None
    best_radius = np.pi

    for init_pts in initializations:
        if time.time() > deadline:
            break

        # Phase 1: Geometric relaxation
        pts, radius = _geometric_relaxation(init_pts, max_iters=500, deadline=deadline)
        if radius < best_radius:
            best_radius = radius
            best_points = pts.copy()

        # Phase 2: Smooth gradient optimization with L-BFGS-B
        if time.time() < deadline - 30.0:
            pts, radius = _smooth_gradient_optimize(
                best_points, deadline=deadline,
                beta_start=15.0, beta_end=300.0, max_stages=4
            )
            if radius < best_radius:
                best_radius = radius
                best_points = pts.copy()

        # Phase 3: Final geometric polish with small steps
        if time.time() < deadline - 10.0:
            pts, radius = _geometric_relaxation(
                best_points, max_iters=300, deadline=deadline,
                step_start=0.02, step_end=0.002
            )
            if radius < best_radius:
                best_radius = radius
                best_points = pts.copy()

    # Final verification
    if best_points is None:
        best_points = _fibonacci_initialization()

    if np.all(np.isfinite(best_points)) and np.all(np.linalg.norm(best_points, axis=1) > 1e-9):
        return _normalize_points(best_points)
    else:
        return _normalize_points(_fibonacci_initialization())
