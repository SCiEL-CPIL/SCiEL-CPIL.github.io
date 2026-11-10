import numpy as np
import time
from scipy.spatial import ConvexHull
import math

N = 130

def _compute_covering_radius(points):
    """
    Computes the covering radius (max circumradius of hull facets) in degrees.
    Returns (radius_in_degrees, worst_facet_index, worst_circumcenter, hull).
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        return 90.0, -1, None, None

    max_radius = 0.0
    worst_facet_idx = -1
    worst_center = None

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

        if R > max_radius:
            max_radius = R
            worst_center = C
            worst_facet_idx = i

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

def _lloyd_iteration(points, max_iter=100):
    """
    Lloyd's algorithm for sphere covering: iteratively move each point to the
    circumcenter of its largest adjacent facet (or the centroid of its Voronoi cell).
    This helps distribute points more evenly.
    """
    points = points.copy()
    for _ in range(max_iter):
        try:
            hull = ConvexHull(points)
        except Exception:
            break
        
        # For each point, find the worst facet it participates in and move towards its circumcenter
        # More efficient: move each point towards the average of circumcenters of facets it touches
        # But simpler: for each facet, move its vertices slightly towards the circumcenter
        # We'll do a weighted average approach
        
        new_points = points.copy()
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
            R = math.acos(cos_R)
            
            C = x[0] * u + x[1] * v + x[2] * w
            norm_C = np.linalg.norm(C)
            if norm_C < 1e-10:
                continue
            C = C / norm_C
            
            # Move vertices towards circumcenter proportional to the radius
            step = 0.05 * R
            for idx in simplex:
                new_points[idx] = _move_point_towards(points[idx], C, step)
        
        # Normalize
        new_points = new_points / np.linalg.norm(new_points, axis=1, keepdims=True)
        
        # Check for improvement
        old_radius, _, _, _ = _compute_covering_radius(points)
        new_radius, _, _, _ = _compute_covering_radius(new_points)
        
        if new_radius < old_radius:
            points = new_points
        else:
            break
    
    return points

def _spectral_optimization(points, max_iter=50):
    """
    Spectral optimization: use the eigenvectors of a matrix designed to maximize
    minimum pairwise separation. We construct a matrix from pairwise dot products
    and use eigenvalue optimization to find better configurations.
    """
    points = points.copy()
    
    for _ in range(max_iter):
        # Compute pairwise dot products
        G = points @ points.T  # Gram matrix
        
        # We want to minimize the maximum angular distance, which is equivalent to
        # maximizing the minimum pairwise angular separation (approximately).
        # Use a spectral approach: find eigenvectors of a modified Gram matrix.
        
        # Construct a matrix that penalizes close points
        # The idea: maximize the minimum eigenvalue of (I - G) or similar
        # Actually, let's use a different approach: iterative refinement based on
        # the worst facets
        
        radius, facet_idx, center, hull = _compute_covering_radius(points)
        if center is None or facet_idx == -1:
            break
        
        simplex = hull.simplices[facet_idx]
        v0, v1, v2 = simplex
        
        # Move the worst facet's vertices towards the circumcenter
        step = 0.03
        new_points = points.copy()
        new_points[v0] = _move_point_towards(points[v0], center, step)
        new_points[v1] = _move_point_towards(points[v1], center, step)
        new_points[v2] = _move_point_towards(points[v2], center, step)
        
        new_radius, _, _, _ = _compute_covering_radius(new_points)
        if new_radius < radius:
            points = new_points
        else:
            # Try smaller steps
            for mult in [0.5, 0.25, 0.1]:
                step = 0.03 * mult
                new_points = points.copy()
                new_points[v0] = _move_point_towards(points[v0], center, step)
                new_points[v1] = _move_point_towards(points[v1], center, step)
                new_points[v2] = _move_point_towards(points[v2], center, step)
                new_radius, _, _, _ = _compute_covering_radius(new_points)
                if new_radius < radius:
                    points = new_points
                    break
    
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    np.random.seed(42)
    
    best_points = None
    best_radius = 90.0
    
    # Phase 1: Global optimization via repulsion from multiple starts
    for seed in range(8):
        if time.time() > deadline - 40:
            break
        
        np.random.seed(seed * 7919 + 42)
        
        # Start with Fibonacci spiral plus jitter
        points = _fibonacci_spiral(N).copy()
        noise = np.random.randn(N, 3) * (0.02 + 0.01 * seed)
        points = points + noise
        points = points / np.linalg.norm(points, axis=1, keepdims=True)
        
        # Global repulsion to spread points
        alpha = 0.03
        for step in range(150):
            if time.time() > deadline - 40:
                break
            
            points = _repulsion_step(points, alpha)
            alpha *= 0.995
            
            # Every 15 steps, check the radius
            if step % 15 == 0 or step == 149:
                radius, _, _, _ = _compute_covering_radius(points)
                if radius < best_radius:
                    best_radius = radius
                    best_points = points.copy()
    
    # Phase 2: Lloyd's iteration for better distribution
    if best_points is not None:
        points = best_points.copy()
    else:
        points = _fibonacci_spiral(N)
        points = points / np.linalg.norm(points, axis=1, keepdims=True)
        best_radius, _, _, _ = _compute_covering_radius(points)
        best_points = points.copy()
    
    if time.time() < deadline - 30:
        points = _lloyd_iteration(points, max_iter=50)
        radius, _, _, _ = _compute_covering_radius(points)
        if radius < best_radius:
            best_radius = radius
            best_points = points.copy()
    
    # Phase 3: Local search with adaptive steps
    if best_points is not None:
        points = best_points.copy()
    
    alpha = 0.08
    stagnation_counter = 0
    max_stagnation = 40
    
    while time.time() < deadline - 15:
        radius, facet_idx, center, hull = _compute_covering_radius(points)
        
        if radius < best_radius:
            best_radius = radius
            best_points = points.copy()
            stagnation_counter = 0
        else:
            stagnation_counter += 1
        
        if center is None or facet_idx == -1:
            break
        
        simplex = hull.simplices[facet_idx]
        v0, v1, v2 = simplex
        
        improved = False
        
        # Try moving all 3 vertices of the worst facet towards its circumcenter
        for step_mult in [1.0, 0.5, 0.25, 0.1]:
            step = alpha * step_mult
            new_points = points.copy()
            new_points[v0] = _move_point_towards(points[v0], center, step)
            new_points[v1] = _move_point_towards(points[v1], center, step)
            new_points[v2] = _move_point_towards(points[v2], center, step)
            
            new_radius, _, _, _ = _compute_covering_radius(new_points)
            if new_radius < radius:
                points = new_points
                improved = True
                break
        
        if not improved:
            # Try moving vertices individually
            for v_idx in [v0, v1, v2]:
                for step_mult in [1.0, 0.5, 0.25]:
                    step = alpha * step_mult
                    test_points = points.copy()
                    test_points[v_idx] = _move_point_towards(points[v_idx], center, step)
                    test_radius, _, _, _ = _compute_covering_radius(test_points)
                    if test_radius < radius:
                        points = test_points
                        improved = True
                        break
                if improved:
                    break
        
        if not improved:
            alpha *= 0.8
            if alpha < 0.003:
                # Perturb the worst facet's vertices to escape local minimum
                noise = np.random.randn(3, 3) * 0.05
                new_points = points.copy()
                new_points[v0] = points[v0] + noise[0]
                new_points[v1] = points[v1] + noise[1]
                new_points[v2] = points[v2] + noise[2]
                
                for idx in [v0, v1, v2]:
                    new_points[idx] = new_points[idx] / np.linalg.norm(new_points[idx])
                
                pert_radius, _, _, _ = _compute_covering_radius(new_points)
                
                if pert_radius < best_radius * 1.02:
                    points = new_points
                    alpha = 0.06
                    stagnation_counter = 0
                else:
                    if stagnation_counter > max_stagnation:
                        break
                    else:
                        alpha = 0.02
    
    # Phase 4: High-precision polish pass
    if time.time() < deadline - 5 and best_points is not None:
        points = best_points.copy()
        polish_alpha = 0.02
        for polish_iter in range(80):
            if time.time() > deadline - 5:
                break
            
            radius, facet_idx, center, hull = _compute_covering_radius(points)
            if center is None or facet_idx == -1:
                break
            
            simplex = hull.simplices[facet_idx]
            v0, v1, v2 = simplex
            
            improved = False
            for step_mult in [1.0, 0.5, 0.25, 0.1, 0.05, 0.02]:
                step = polish_alpha * step_mult
                new_points = points.copy()
                new_points[v0] = _move_point_towards(points[v0], center, step)
                new_points[v1] = _move_point_towards(points[v1], center, step)
                new_points[v2] = _move_point_towards(points[v2], center, step)
                
                new_radius, _, _, _ = _compute_covering_radius(new_points)
                if new_radius < radius:
                    points = new_points
                    improved = True
                    break
            
            if not improved:
                for v_idx in [v0, v1, v2]:
                    for step_mult in [1.0, 0.5, 0.25, 0.1, 0.05]:
                        step = polish_alpha * step_mult
                        test_points = points.copy()
                        test_points[v_idx] = _move_point_towards(points[v_idx], center, step)
                        test_radius, _, _, _ = _compute_covering_radius(test_points)
                        if test_radius < radius:
                            points = test_points
                            improved = True
                            break
                    if improved:
                        break
            
            if not improved:
                polish_alpha *= 0.65
                if polish_alpha < 0.0003:
                    break
            else:
                if radius < best_radius:
                    best_radius = radius
                    best_points = points.copy()
    
    # Phase 5: Final fine-tuning
    if time.time() < deadline - 2 and best_points is not None:
        points = best_points.copy()
        for final_iter in range(30):
            if time.time() > deadline - 2:
                break
            
            radius, facet_idx, center, hull = _compute_covering_radius(points)
            if center is None or facet_idx == -1:
                break
            
            simplex = hull.simplices[facet_idx]
            v0, v1, v2 = simplex
            
            improved = False
            for step in [0.005, 0.002, 0.001, 0.0005]:
                new_points = points.copy()
                new_points[v0] = _move_point_towards(points[v0], center, step)
                new_points[v1] = _move_point_towards(points[v1], center, step)
                new_points[v2] = _move_point_towards(points[v2], center, step)
                
                new_radius, _, _, _ = _compute_covering_radius(new_points)
                if new_radius < radius:
                    points = new_points
                    improved = True
                    break
            
            if not improved:
                break
            else:
                if radius < best_radius:
                    best_radius = radius
                    best_points = points.copy()
    
    return best_points
