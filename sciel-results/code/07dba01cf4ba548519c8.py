import numpy as np
import time
from scipy.spatial import ConvexHull
import math

N = 130

def _compute_all_radii(points, hull):
    """
    Compute circumradius and circumcenter for all hull facets.
    Returns arrays: radii (n_facets,), centers (n_facets, 3), valid_mask (n_facets,)
    """
    n_facets = len(hull.simplices)
    radii = np.zeros(n_facets)
    centers = np.zeros((n_facets, 3))
    valid = np.zeros(n_facets, dtype=bool)

    for i, simplex in enumerate(hull.simplices):
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

        C = x[0] * u + x[1] * v + x[2] * w
        norm_C = np.linalg.norm(C)
        if norm_C < 1e-10:
            continue
        C = C / norm_C

        R = math.acos(cos_R)
        radii[i] = R
        centers[i] = C
        valid[i] = True

    return radii, centers, valid

def _compute_covering_radius(points):
    """
    Computes the covering radius (max circumradius of hull facets) in degrees.
    Returns (radius_in_degrees, worst_facet_index, worst_circumcenter, hull).
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        return 90.0, -1, None, None

    radii, centers, valid = _compute_all_radii(points, hull)
    
    if not np.any(valid):
        return 90.0, -1, None, None

    valid_indices = np.where(valid)[0]
    valid_radii = radii[valid_indices]
    
    max_idx_in_valid = np.argmax(valid_radii)
    worst_facet_idx = valid_indices[max_idx_in_valid]
    max_radius = valid_radii[max_idx_in_valid]
    worst_center = centers[worst_facet_idx]

    return math.degrees(max_radius), worst_facet_idx, worst_center, hull

def _move_point_towards(p, c, step_size):
    """Move point p on the sphere towards c by step_size radians."""
    dot_pc = np.dot(p, c)
    t = c - dot_pc * p
    norm_t = np.linalg.norm(t)
    if norm_t < 1e-10:
        return p
    t = t / norm_t
    new_p = p + step_size * t
    new_p = new_p / np.linalg.norm(new_p)
    return new_p

def _fibonacci_spiral(n):
    """Generate a Fibonacci spiral configuration on the unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return points

def _repulsion_step(points, alpha, iters=1):
    """Perform repulsion steps to spread points out."""
    for _ in range(iters):
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
        points = new_points
    return points

def _smooth_gradient(points, beta=50.0):
    """
    Compute smooth approximation of covering radius gradient.
    Returns (smooth_radius, gradient) where gradient points in direction to DECREASE radius.
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        return 90.0 * math.pi / 180.0, np.zeros_like(points)

    radii, centers, valid = _compute_all_radii(points, hull)
    
    if not np.any(valid):
        return 90.0 * math.pi / 180.0, np.zeros_like(points)

    valid_indices = np.where(valid)[0]
    valid_radii = radii[valid_indices]
    
    max_r = np.max(valid_radii)
    if max_r < 1e-10:
        return 0.0, np.zeros_like(points)

    # Soft-max weights
    shifted = beta * (valid_radii - max_r)
    weights = np.exp(shifted)
    weights = weights / np.sum(weights)

    smooth_radius = max_r + (1.0 / beta) * np.log(np.sum(weights))

    # Gradient: move vertices of high-weight facets towards their circumcenters
    gradient = np.zeros_like(points)
    
    for idx, facet_idx in enumerate(valid_indices):
        w_i = weights[idx]
        if w_i < 1e-8:
            continue
        
        simplex = hull.simplices[facet_idx]
        C = centers[facet_idx]
        
        for v_idx in simplex:
            p = points[v_idx]
            dot_pc = np.dot(p, C)
            t = C - dot_pc * p
            norm_t = np.linalg.norm(t)
            if norm_t > 1e-10:
                t = t / norm_t
                gradient[v_idx] += w_i * t

    # Normalize per-point gradient
    grad_norms = np.linalg.norm(gradient, axis=1, keepdims=True)
    gradient = gradient / np.where(grad_norms == 0, 1, grad_norms)

    return smooth_radius, gradient

def _local_search_facet(points, alpha_init=0.05, max_iters=200, time_limit=None):
    """
    Facet-based local search: iteratively move worst facet vertices toward circumcenter.
    Returns improved points.
    """
    alpha = alpha_init
    stagnation = 0
    max_stagnation = 50
    start = time.time()
    
    for _ in range(max_iters):
        if time_limit is not None and time.time() > time_limit:
            break
        
        radius_deg, facet_idx, center, hull = _compute_covering_radius(points)
        if center is None or facet_idx == -1:
            break
        
        simplex = hull.simplices[facet_idx]
        v0, v1, v2 = simplex
        
        improved = False
        
        # Try moving all 3 vertices together
        new_points = points.copy()
        new_points[v0] = _move_point_towards(points[v0], center, alpha)
        new_points[v1] = _move_point_towards(points[v1], center, alpha)
        new_points[v2] = _move_point_towards(points[v2], center, alpha)
        
        new_radius, _, _, _ = _compute_covering_radius(new_points)
        
        if new_radius < radius_deg - 1e-12:
            points = new_points
            improved = True
            stagnation = 0
        else:
            # Try individual vertices
            for v_idx in [v0, v1, v2]:
                test_points = points.copy()
                test_points[v_idx] = _move_point_towards(points[v_idx], center, alpha)
                test_radius, _, _, _ = _compute_covering_radius(test_points)
                if test_radius < radius_deg - 1e-12:
                    points = test_points
                    improved = True
                    stagnation = 0
                    break
        
        if not improved:
            alpha *= 0.8
            stagnation += 1
            if alpha < 0.003:
                # Perturb to escape
                noise = np.random.randn(3, 3) * 0.03
                new_points = points.copy()
                new_points[v0] = points[v0] + noise[0]
                new_points[v1] = points[v1] + noise[1]
                new_points[v2] = points[v2] + noise[2]
                for idx in [v0, v1, v2]:
                    new_points[idx] = new_points[idx] / np.linalg.norm(new_points[idx])
                
                pert_radius, _, _, _ = _compute_covering_radius(new_points)
                if pert_radius < radius_deg + 0.5:
                    points = new_points
                    alpha = alpha_init
                    stagnation = 0
                elif stagnation > max_stagnation:
                    break
                else:
                    alpha = 0.02
        else:
            alpha = min(alpha * 1.01, alpha_init)
    
    return points

def _polish_pass(points, time_limit):
    """High-precision polish with very small steps."""
    alpha = 0.005
    for _ in range(500):
        if time.time() > time_limit:
            break
        
        radius_deg, facet_idx, center, hull = _compute_covering_radius(points)
        if center is None or facet_idx == -1:
            break
        
        simplex = hull.simplices[facet_idx]
        v0, v1, v2 = simplex
        
        improved = False
        for mult in [1.0, 0.5, 0.25, 0.1, 0.05]:
            step = alpha * mult
            new_points = points.copy()
            new_points[v0] = _move_point_towards(points[v0], center, step)
            new_points[v1] = _move_point_towards(points[v1], center, step)
            new_points[v2] = _move_point_towards(points[v2], center, step)
            
            new_radius, _, _, _ = _compute_covering_radius(new_points)
            if new_radius < radius_deg - 1e-14:
                points = new_points
                improved = True
                break
        
        if not improved:
            # Try single vertex moves
            for v_idx in [v0, v1, v2]:
                for mult in [1.0, 0.5, 0.25, 0.1]:
                    step = alpha * mult
                    test_points = points.copy()
                    test_points[v_idx] = _move_point_towards(points[v_idx], center, step)
                    test_radius, _, _, _ = _compute_covering_radius(test_points)
                    if test_radius < radius_deg - 1e-14:
                        points = test_points
                        improved = True
                        break
                if improved:
                    break
        
        if not improved:
            alpha *= 0.7
            if alpha < 1e-5:
                break
    
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin

    np.random.seed(42)

    best_points = None
    best_radius = 90.0

    # Phase 1: Global optimization via repulsion from multiple starts
    phase1_deadline = start_time + 45.0
    for seed in range(10):
        if time.time() > phase1_deadline:
            break

        np.random.seed(seed * 1000 + 42)

        points = _fibonacci_spiral(N).copy()
        noise = np.random.randn(N, 3) * 0.02
        points = points + noise
        points = points / np.linalg.norm(points, axis=1, keepdims=True)

        alpha = 0.015
        for step in range(400):
            if time.time() > phase1_deadline:
                break

            points = _repulsion_step(points, alpha, iters=1)

            if step % 40 == 0 or step == 399:
                radius, _, _, _ = _compute_covering_radius(points)
                if radius < best_radius:
                    best_radius = radius
                    best_points = points.copy()

    # Phase 2: Hybrid local search - smooth gradient + facet-based
    if best_points is not None:
        points = best_points.copy()
    else:
        points = _fibonacci_spiral(N)
        points = points / np.linalg.norm(points, axis=1, keepdims=True)
        best_radius, _, _, _ = _compute_covering_radius(points)
        best_points = points.copy()

    phase2_deadline = start_time + 115.0
    alpha = 0.03
    beta = 40.0
    stagnation = 0
    max_stagnation = 80
    use_gradient = True

    while time.time() < phase2_deadline:
        radius_deg, facet_idx, center, hull = _compute_covering_radius(points)
        
        if radius_deg < best_radius:
            best_radius = radius_deg
            best_points = points.copy()
            stagnation = 0
        else:
            stagnation += 1

        if center is None or facet_idx == -1:
            break

        if use_gradient and stagnation < 30:
            # Use smooth gradient for broader moves
            _, gradient = _smooth_gradient(points, beta)
            
            new_points = points + alpha * gradient
            new_points = new_points / np.linalg.norm(new_points, axis=1, keepdims=True)
            
            new_radius, _, _, _ = _compute_covering_radius(new_points)
            
            if new_radius < radius_deg - 1e-10:
                points = new_points
                alpha *= 0.97
                beta = min(beta * 1.03, 120.0)
                stagnation = 0
            else:
                # Try smaller gradient step
                improved = False
                for mult in [0.5, 0.25, 0.1]:
                    test_points = points + alpha * mult * gradient
                    test_points = test_points / np.linalg.norm(test_points, axis=1, keepdims=True)
                    test_radius, _, _, _ = _compute_covering_radius(test_points)
                    if test_radius < radius_deg - 1e-10:
                        points = test_points
                        alpha *= 0.95
                        improved = True
                        break
                
                if not improved:
                    alpha *= 0.8
                    if alpha < 0.005:
                        use_gradient = False
                        alpha = 0.04
                        beta = 40.0
                        stagnation = 0
        else:
            # Facet-based local search
            simplex = hull.simplices[facet_idx]
            v0, v1, v2 = simplex
            
            improved = False
            
            # Move all 3 vertices
            new_points = points.copy()
            new_points[v0] = _move_point_towards(points[v0], center, alpha)
            new_points[v1] = _move_point_towards(points[v1], center, alpha)
            new_points[v2] = _move_point_towards(points[v2], center, alpha)
            
            new_radius, _, _, _ = _compute_covering_radius(new_points)
            
            if new_radius < radius_deg - 1e-10:
                points = new_points
                improved = True
                alpha = min(alpha * 1.02, 0.06)
            else:
                # Individual vertex moves
                for v_idx in [v0, v1, v2]:
                    test_points = points.copy()
                    test_points[v_idx] = _move_point_towards(points[v_idx], center, alpha)
                    test_radius, _, _, _ = _compute_covering_radius(test_points)
                    if test_radius < radius_deg - 1e-10:
                        points = test_points
                        improved = True
                        break
            
            if not improved:
                alpha *= 0.85
                if alpha < 0.004:
                    # Perturb
                    noise = np.random.randn(3, 3) * 0.04
                    new_points = points.copy()
                    new_points[v0] = points[v0] + noise[0]
                    new_points[v1] = points[v1] + noise[1]
                    new_points[v2] = points[v2] + noise[2]
                    for idx in [v0, v1, v2]:
                        new_points[idx] = new_points[idx] / np.linalg.norm(new_points[idx])
                    
                    pert_radius, _, _, _ = _compute_covering_radius(new_points)
                    if pert_radius < best_radius * 1.02:
                        points = new_points
                        alpha = 0.04
                        use_gradient = True
                        beta = 40.0
                        stagnation = 0
                    elif stagnation > max_stagnation:
                        break
                    else:
                        alpha = 0.02
    
    # Phase 3: Polish pass
    if time.time() < deadline - 3:
        points = best_points.copy()
        polished = _polish_pass(points, deadline - 2)
        polished_radius, _, _, _ = _compute_covering_radius(polished)
        if polished_radius < best_radius:
            best_radius = polished_radius
            best_points = polished

    # Final validation
    if best_points is None:
        best_points = _fibonacci_spiral(N)
        best_points = best_points / np.linalg.norm(best_points, axis=1, keepdims=True)

    # Ensure all points are on unit sphere
    best_points = best_points / np.linalg.norm(best_points, axis=1, keepdims=True)
    
    return best_points
