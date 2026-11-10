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

def _vectorized_repulsion(points, steps=50, alpha_start=0.05, decay=0.99):
    """
    Perform vectorized repulsion steps to spread points.
    """
    pts = points.copy()
    alpha = alpha_start
    for _ in range(steps):
        # Compute pairwise differences
        # diff[i, j] = pts[i] - pts[j]
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff ** 2, axis=2)
        
        # Avoid division by zero for i==j
        np.fill_diagonal(dist_sq, 1.0)
        
        # Force magnitude: 1 / dist^3 (Coulomb-like)
        # We want to push apart. Force on i from j is (pts[i] - pts[j]) / |pts[i] - pts[j]|^3
        # Actually, standard repulsion is (pts[i] - pts[j]) / |pts[i] - pts[j]|^3
        # Let's use 1/dist^2 for stability or 1/dist^3.
        # Using 1/dist^3 is standard for sphere packing repulsion.
        
        # dist_sq is |diff|^2. dist is sqrt(dist_sq).
        # factor = 1 / dist^3 = 1 / (dist_sq * sqrt(dist_sq))
        with np.errstate(divide='ignore', invalid='ignore'):
            factor = 1.0 / (dist_sq * np.sqrt(dist_sq))
        
        # Zero out diagonal
        np.fill_diagonal(factor, 0.0)
        
        # Force on i is sum over j of factor[i,j] * (pts[i] - pts[j])
        # force[i] = sum_j factor[i,j] * diff[i,j]
        forces = np.sum(factor[:, :, np.newaxis] * diff, axis=1)
        
        # Normalize forces to prevent huge jumps
        norms = np.linalg.norm(forces, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        forces = forces / norms
        
        # Update points
        pts = pts + alpha * forces
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        
        alpha *= decay
        
    return pts

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

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin

    # Phase 1: Initialization and Repulsion
    # Use a few different initializations to escape local minima
    best_points = None
    best_radius = 90.0

    initializations = [
        _fibonacci_spiral(N),
        _fibonacci_spiral(N) + np.random.randn(N, 3) * 0.05,
        _fibonacci_spiral(N) + np.random.randn(N, 3) * 0.1
    ]
    
    for init in initializations:
        if time.time() > deadline - 60:
            break
        
        # Normalize
        pts = init / np.linalg.norm(init, axis=1, keepdims=True)
        
        # Repulsion
        pts = _vectorized_repulsion(pts, steps=50, alpha_start=0.05, decay=0.99)
        
        radius, _, _ = _compute_covering_radius(pts)
        if radius < best_radius:
            best_radius = radius
            best_points = pts.copy()

    if best_points is None:
        best_points = _fibonacci_spiral(N)
        best_radius, _, _ = _compute_covering_radius(best_points)

    # Phase 2: Local Search (Worst Facet Focusing)
    points = best_points.copy()
    current_radius = best_radius
    
    # Parameters for local search
    base_step = 0.05
    step_decay = 0.95
    stagnation_counter = 0
    max_stagnation = 20
    step = base_step

    while time.time() < deadline - 5:
        radius, facet_idx, hull = _compute_covering_radius(points)
        
        if radius < current_radius:
            current_radius = radius
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
        
        # Strategy 1: Move all 3 vertices towards circumcenter
        new_points = points.copy()
        new_points[v0] = _move_point_towards(points[v0], C, step)
        new_points[v1] = _move_point_towards(points[v1], C, step)
        new_points[v2] = _move_point_towards(points[v2], C, step)
        new_radius, _, _ = _compute_covering_radius(new_points)
        if new_radius < radius:
            points = new_points
            improved = True
        else:
            # Strategy 2: Move individual vertices towards circumcenter
            for v_idx in [v0, v1, v2]:
                test_points = points.copy()
                test_points[v_idx] = _move_point_towards(points[v_idx], C, step)
                test_radius, _, _ = _compute_covering_radius(test_points)
                if test_radius < radius:
                    points = test_points
                    improved = True
                    break
            
            # Strategy 3: Random perturbation if no improvement
            if not improved:
                for _ in range(3):
                    test_points = points.copy()
                    for v_idx in [v0, v1, v2]:
                        noise = np.random.randn(3) * step * 0.5
                        test_points[v_idx] = test_points[v_idx] + noise
                        test_points[v_idx] = test_points[v_idx] / np.linalg.norm(test_points[v_idx])
                    test_radius, _, _ = _compute_covering_radius(test_points)
                    if test_radius < radius:
                        points = test_points
                        improved = True
                        break

        if not improved:
            step *= step_decay
            if step < 0.001:
                # Reset step and perturb more strongly
                step = base_step
                # Random restart of the local search region
                if stagnation_counter > max_stagnation:
                    break
        else:
            step = base_step * 0.8 # Slightly reduce step on success to refine

    return best_points
