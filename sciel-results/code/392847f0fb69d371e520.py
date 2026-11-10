import numpy as np
import time
from scipy.spatial import ConvexHull
import math

N = 130

def _compute_covering_radius(points):
    """
    Computes the covering radius (max circumradius of hull facets) in degrees.
    Returns (radius_in_degrees, worst_facet_index, hull).
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        return 90.0, -1, None

    max_radius = 0.0
    worst_facet_idx = -1
    n_simplices = len(hull.simplices)

    for i in range(n_simplices):
        simplex = hull.simplices[i]
        u = points[simplex[0]]
        v = points[simplex[1]]
        w = points[simplex[2]]

        uv = np.dot(u, v)
        uw = np.dot(u, w)
        vw = np.dot(v, w)

        M = np.array([
            [1.0, uv, uw],
            [uv, 1.0, vw],
            [uw, vw, 1.0]
        ])

        try:
            x = np.linalg.solve(M, np.ones(3))
        except np.linalg.LinAlgError:
            continue

        xT_M_x = np.dot(x, M @ x)
        if xT_M_x <= 1e-10:
            continue

        cos_R = 1.0 / np.sqrt(xT_M_x)
        cos_R = np.clip(cos_R, -1.0, 1.0)
        R = math.acos(cos_R)

        if R > max_radius:
            max_radius = R
            worst_facet_idx = i

    return math.degrees(max_radius), worst_facet_idx, hull

def _fibonacci_spiral(n):
    """Generate a Fibonacci spiral configuration on the unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return points

def _repulsion_step(points, alpha):
    """Perform one repulsion step to spread points out."""
    forces = np.zeros_like(points)
    for i in range(N):
        for j in range(i + 1, N):
            diff = points[i] - points[j]
            dist_sq = np.dot(diff, diff)
            if dist_sq < 1e-12:
                continue
            force = diff / (dist_sq ** 1.5)
            forces[i] += force
            forces[j] -= force

    forces_norm = np.linalg.norm(forces, axis=1, keepdims=True)
    forces = forces / np.where(forces_norm == 0, 1, forces_norm)

    new_points = points + alpha * forces
    new_points = new_points / np.linalg.norm(new_points, axis=1, keepdims=True)
    return new_points

def _move_point_towards(p, c, step_size):
    """Move point p on the sphere towards c by step_size radians."""
    dot_pc = np.dot(p, c)
    t = c - dot_pc * p
    norm_t = np.linalg.norm(t)
    if norm_t < 1e-10:
        return p.copy()
    t = t / norm_t
    new_p = p + step_size * t
    new_p = new_p / np.linalg.norm(new_p)
    return new_p

def _spectral_init(n, n_iter=1000):
    """
    Initialize points using a spectral/repulsion hybrid.
    Start from Fibonacci, then optimize via repulsion with decreasing alpha.
    """
    points = _fibonacci_spiral(n).copy()
    
    alpha = 0.05
    for it in range(n_iter):
        alpha = 0.05 * (0.995 ** it)
        if alpha < 1e-6:
            break
        points = _repulsion_step(points, alpha)
    
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin

    np.random.seed(42)

    best_points = None
    best_radius = 90.0

    # Phase 1: Multiple starts with repulsion
    for seed in range(8):
        if time.time() > deadline - 40:
            break

        np.random.seed(seed * 1000 + 42)

        if seed == 0:
            points = _spectral_init(N, n_iter=500)
        else:
            # Different initializations
            points = _fibonacci_spiral(N).copy()
            noise = np.random.randn(N, 3) * (0.01 + 0.02 * seed / 7)
            points = points + noise
            points = points / np.linalg.norm(points, axis=1, keepdims=True)
            
            # Short repulsion
            alpha = 0.03
            for step in range(100):
                if time.time() > deadline - 40:
                    break
                alpha = 0.03 * (0.99 ** step)
                if alpha < 1e-5:
                    break
                points = _repulsion_step(points, alpha)

        radius, _, _ = _compute_covering_radius(points)
        if radius < best_radius:
            best_radius = radius
            best_points = points.copy()

    if best_points is None:
        best_points = _fibonacci_spiral(N)
        best_radius, _, _ = _compute_covering_radius(best_points)

    # Phase 2: Local search - move worst facet vertices towards circumcenter
    points = best_points.copy()
    alpha = 0.08
    stagnation_counter = 0
    max_stagnation = 50
    best_local_radius = best_radius

    while time.time() < deadline - 15:
        radius, facet_idx, hull = _compute_covering_radius(points)

        if radius < best_local_radius:
            best_local_radius = radius
            if radius < best_radius:
                best_radius = radius
                best_points = points.copy()
            stagnation_counter = 0
        else:
            stagnation_counter += 1

        if facet_idx == -1:
            break

        simplex = hull.simplices[facet_idx]
        v0, v1, v2 = simplex

        # Compute circumcenter of worst facet
        u = points[v0]
        v = points[v1]
        w = points[v2]
        uv = np.dot(u, v)
        uw = np.dot(u, w)
        vw = np.dot(v, w)
        M = np.array([
            [1.0, uv, uw],
            [uv, 1.0, vw],
            [uw, vw, 1.0]
        ])
        try:
            x = np.linalg.solve(M, np.ones(3))
        except np.linalg.LinAlgError:
            break
        xT_M_x = np.dot(x, M @ x)
        if xT_M_x <= 1e-10:
            break
        C = x[0] * u + x[1] * v + x[2] * w
        norm_C = np.linalg.norm(C)
        if norm_C < 1e-10:
            break
        C = C / norm_C

        improved = False

        # Try moving all 3 vertices towards circumcenter
        new_points = points.copy()
        new_points[v0] = _move_point_towards(points[v0], C, alpha)
        new_points[v1] = _move_point_towards(points[v1], C, alpha)
        new_points[v2] = _move_point_towards(points[v2], C, alpha)
        new_radius, _, _ = _compute_covering_radius(new_points)
        if new_radius < radius:
            points = new_points
            improved = True
        else:
            # Try individual vertex moves
            for v_idx in [v0, v1, v2]:
                test_points = points.copy()
                test_points[v_idx] = _move_point_towards(points[v_idx], C, alpha)
                test_radius, _, _ = _compute_covering_radius(test_points)
                if test_radius < radius:
                    points = test_points
                    improved = True
                    break

        if not improved:
            alpha *= 0.85
            if alpha < 0.003:
                # Perturb to escape local minimum
                noise = np.random.randn(3, 3) * 0.02
                new_points = points.copy()
                new_points[v0] = points[v0] + noise[0]
                new_points[v1] = points[v1] + noise[1]
                new_points[v2] = points[v2] + noise[2]
                for idx in [v0, v1, v2]:
                    new_points[idx] = new_points[idx] / np.linalg.norm(new_points[idx])
                pert_radius, _, _ = _compute_covering_radius(new_points)
                if pert_radius < best_local_radius * 1.01:
                    points = new_points
                    alpha = 0.05
                    stagnation_counter = 0
                else:
                    if stagnation_counter > max_stagnation:
                        break
                    alpha = 0.015

    # Phase 3: High-precision polish
    if time.time() < deadline - 3:
        points = best_points.copy()
        polish_alpha = 0.02
        for polish_iter in range(100):
            if time.time() > deadline - 3:
                break

            radius, facet_idx, hull = _compute_covering_radius(points)
            if facet_idx == -1:
                break

            simplex = hull.simplices[facet_idx]
            v0, v1, v2 = simplex

            # Compute circumcenter
            u = points[v0]
            v = points[v1]
            w = points[v2]
            uv = np.dot(u, v)
            uw = np.dot(u, w)
            vw = np.dot(v, w)
            M = np.array([
                [1.0, uv, uw],
                [uv, 1.0, vw],
                [uw, vw, 1.0]
            ])
            try:
                x = np.linalg.solve(M, np.ones(3))
            except np.linalg.LinAlgError:
                break
            xT_M_x = np.dot(x, M @ x)
            if xT_M_x <= 1e-10:
                break
            C = x[0] * u + x[1] * v + x[2] * w
            norm_C = np.linalg.norm(C)
            if norm_C < 1e-10:
                break
            C = C / norm_C

            improved = False
            for step_mult in [1.0, 0.5, 0.25, 0.1, 0.05]:
                step = polish_alpha * step_mult
                new_points = points.copy()
                new_points[v0] = _move_point_towards(points[v0], C, step)
                new_points[v1] = _move_point_towards(points[v1], C, step)
                new_points[v2] = _move_point_towards(points[v2], C, step)
                new_radius, _, _ = _compute_covering_radius(new_points)
                if new_radius < radius:
                    points = new_points
                    improved = True
                    break

            if not improved:
                # Try individual moves
                for v_idx in [v0, v1, v2]:
                    for step_mult in [1.0, 0.5, 0.25]:
                        step = polish_alpha * step_mult
                        test_points = points.copy()
                        test_points[v_idx] = _move_point_towards(points[v_idx], C, step)
                        test_radius, _, _ = _compute_covering_radius(test_points)
                        if test_radius < radius:
                            points = test_points
                            improved = True
                            break
                    if improved:
                        break

            if not improved:
                polish_alpha *= 0.7
                if polish_alpha < 0.0005:
                    break
            else:
                if radius < best_radius:
                    best_radius = radius
                    best_points = points.copy()

    return best_points
