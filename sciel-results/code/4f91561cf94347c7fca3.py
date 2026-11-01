import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time

N = 130

def _cartesian_to_spherical(points):
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    r = np.sqrt(x**2 + y**2 + z**2)
    r_safe = np.where(r == 0, 1e-9, r)
    theta = np.arccos(np.clip(z / r_safe, -1.0, 1.0))
    phi = np.arctan2(y, x)
    phi = phi % (2 * np.pi)
    return np.column_stack([theta, phi])

def _spherical_to_cartesian(coords):
    theta = coords[:, 0]
    phi = coords[:, 1]
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.column_stack([x, y, z])

def _covering_radius(points, n_grid=20000):
    i = np.arange(n_grid) + 0.5
    z_grid = 1.0 - 2.0 * i / n_grid
    phi_grid = np.pi * (1.0 + np.sqrt(5.0)) * i
    r_grid = np.sqrt(1.0 - z_grid * z_grid)
    grid_points = np.column_stack([r_grid * np.cos(phi_grid), r_grid * np.sin(phi_grid), z_grid])
    
    from scipy.spatial.distance import cdist
    dists = cdist(grid_points, points)
    min_dists = np.min(dists, axis=1)
    min_dists_clipped = np.clip(min_dists, 0.0, 2.0)
    angular_dists = 2.0 * np.arcsin(min_dists_clipped / 2.0)
    return np.max(angular_dists)

def _exact_covering_radius(points):
    """
    Compute the exact covering radius from the convex hull.
    The covering radius is the maximum circumradius of the facets (spherical triangles)
    of the convex hull, where each facet is checked to have an empty cap.
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi
    
    max_radius = 0.0
    for simplex in hull.simplices:
        verts = points[simplex]
        a, b, c = verts[0], verts[1], verts[2]
        normal = np.cross(b - a, c - a)
        norm_val = np.linalg.norm(normal)
        if norm_val < 1e-12:
            continue
        normal = normal / norm_val
        
        # Ensure normal points outward
        centroid = np.mean(verts, axis=0)
        if np.dot(normal, centroid) < 0:
            normal = -normal
        
        # The circumcenter of the spherical triangle is in the direction of the normal
        # The angular distance from the normal to each vertex is the angle between vertex and normal
        dots = np.clip(np.dot(verts, normal), -1.0, 1.0)
        angles = np.arccos(dots)
        facet_radius = np.max(angles)
        max_radius = max(max_radius, facet_radius)
    
    return max_radius

def _soft_covering_radius_and_grad(points, grid_points, sigma=0.05):
    """
    Vectorized soft covering radius and its gradient.
    Uses soft-min over points for each grid point, then soft-max over grid points.
    """
    dots = grid_points @ points.T
    dots = np.clip(dots, -1.0, 1.0)
    angles = np.arccos(dots)
    
    # Soft-min: weighted average of angles using softmax weights over negative angles
    neg_scaled = -angles / sigma
    max_neg = np.max(neg_scaled, axis=1, keepdims=True)
    exp_neg = np.exp(neg_scaled - max_neg)
    weights = exp_neg / np.sum(exp_neg, axis=1, keepdims=True)
    
    # Soft-min value
    soft_min_vals = np.sum(weights * angles, axis=1)
    
    # Soft-max of soft-min values
    scaled_max = soft_min_vals / sigma
    max_scaled = np.max(scaled_max)
    exp_max = np.exp(scaled_max - max_scaled)
    max_weights = exp_max / np.sum(exp_max)
    soft_max_val = sigma * (max_scaled + np.log(np.sum(exp_max)))
    
    # Gradient computation
    denom = np.sqrt(np.maximum(1.0 - dots * dots, 1e-18))
    
    coeff = max_weights[:, np.newaxis] * weights
    alpha = coeff / denom
    
    # Term 1: -sum_i alpha[i,j] * g_i
    term1 = -(alpha.T @ grid_points)
    
    # Term 2: sum_i alpha[i,j] * dots[i,j] * p_j
    beta = alpha * dots
    beta_sum = np.sum(beta, axis=0)
    term2 = beta_sum[:, np.newaxis] * points
    
    grad = term1 + term2
    
    return soft_max_val, grad

def _lloyd_step(points, grid_points, n_iters=3):
    """
    Perform Lloyd's algorithm iterations using grid-based Voronoi.
    """
    for _ in range(n_iters):
        dots = grid_points @ points.T
        angles = np.arccos(np.clip(dots, -1.0, 1.0))
        nearest = np.argmin(angles, axis=1)
        
        new_points = np.zeros_like(points)
        
        # Vectorized: sum grid points per cluster
        for j in range(N):
            mask = (nearest == j)
            count = np.sum(mask)
            if count > 0:
                new_points[j] = np.sum(grid_points[mask], axis=0) / count
            else:
                new_points[j] = points[j]
        
        norms = np.linalg.norm(new_points, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        new_points = new_points / norms
        
        diff = np.linalg.norm(new_points - points, axis=1)
        points = new_points
        
        if np.max(diff) < 1e-7:
            break
    
    return points

def _facet_gradient_descent(points, max_iters=50, step_size=0.01):
    """
    Gradient descent that moves vertices toward their facet's circumcenter
    to reduce the circumradius of the worst facets.
    """
    pts = points.copy()
    for it in range(max_iters):
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
        
        # Compute gradient: for each point, move it toward the circumcenter of its worst facet
        grad = np.zeros_like(pts)
        
        for simplex in hull.simplices:
            verts = pts[simplex]
            a, b, c = verts[0], verts[1], verts[2]
            normal = np.cross(b - a, c - a)
            norm_val = np.linalg.norm(normal)
            if norm_val < 1e-12:
                continue
            normal = normal / norm_val
            
            centroid = np.mean(verts, axis=0)
            if np.dot(normal, centroid) < 0:
                normal = -normal
            
            dots = np.clip(np.dot(verts, normal), -1.0, 1.0)
            angles = np.arccos(dots)
            facet_radius = np.max(angles)
            
            # Move each vertex toward the normal direction (circumcenter)
            # The direction to move is (normal - vertex) projected to tangent
            for idx, vert in enumerate(verts):
                # Tangent direction toward normal
                tangent_dir = normal - np.dot(vert, normal) * vert
                tangent_dir = tangent_dir / (np.linalg.norm(tangent_dir) + 1e-12)
                # Weight by how much this vertex contributes to the radius
                weight = angles[idx]
                grad[simplex[idx]] += weight * tangent_dir
        
        # Normalize gradient
        grad_norms = np.linalg.norm(grad, axis=1, keepdims=True)
        grad_norms = np.where(grad_norms < 1e-12, 1.0, grad_norms)
        grad = grad / grad_norms
        
        # Update points
        pts = pts + step_size * grad
        
        # Project back to sphere
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        
        # Check convergence
        if np.max(np.abs(grad)) < 1e-6:
            break
    
    return pts

def _optimize_single(start_coords, time_limit=10.0, grid_points=None):
    """
    Optimize a single configuration starting from start_coords.
    Combines soft covering radius gradient descent with Lloyd's algorithm.
    """
    start_time = time.time()
    
    # Convert to Cartesian
    points = _spherical_to_cartesian(start_coords)
    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    
    if grid_points is None:
        i = np.arange(10000) + 0.5
        z_grid = 1.0 - 2.0 * i / 10000
        phi_grid = np.pi * (1.0 + np.sqrt(5.0)) * i
        r_grid = np.sqrt(1.0 - z_grid * z_grid)
        grid_points = np.column_stack([r_grid * np.cos(phi_grid), r_grid * np.sin(phi_grid), z_grid])
    
    # Phase 1: Lloyd's algorithm for coarse optimization
    if time.time() - start_time < time_limit * 0.3:
        points = _lloyd_step(points, grid_points, n_iters=5)
    
    # Phase 2: Soft covering radius gradient descent
    if time.time() - start_time < time_limit * 0.7:
        coords = _cartesian_to_spherical(points)
        x0 = coords.flatten()
        bounds = [(0.0, np.pi), (0.0, 2.0 * np.pi)] * N
        
        def objective(x_flat):
            coords = x_flat.reshape(N, 2)
            pts = _spherical_to_cartesian(coords)
            pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
            val, _ = _soft_covering_radius_and_grad(pts, grid_points, sigma=0.05)
            return val
        
        def jac(x_flat):
            coords = x_flat.reshape(N, 2)
            pts = _spherical_to_cartesian(coords)
            pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
            _, grad_cart = _soft_covering_radius_and_grad(pts, grid_points, sigma=0.05)
            
            # Convert Cartesian gradient to spherical gradient
            # This is complex, so we'll use numerical gradient as fallback
            # For now, just return zeros and let L-BFGS-B compute numerically
            return None
        
        try:
            res = minimize(
                objective, 
                x0, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-6, 'disp': False}
            )
            if res.success or res.status in [1, 2]:
                coords = res.x.reshape(N, 2)
                points = _spherical_to_cartesian(coords)
                points = points / np.linalg.norm(points, axis=1, keepdims=True)
        except Exception:
            pass
    
    # Phase 3: Facet-based gradient descent for fine polishing
    if time.time() - start_time < time_limit * 0.9:
        points = _facet_gradient_descent(points, max_iters=30, step_size=0.005)
    
    # Phase 4: Final Lloyd iterations
    if time.time() - start_time < time_limit * 0.95:
        points = _lloyd_step(points, grid_points, n_iters=3)
    
    return points

def construct():
    start_time = time.time()
    total_time_limit = 140.0  # Leave 10s margin for 150s limit
    
    best_score = np.inf
    best_points = None
    
    # Generate grid points for reuse
    i = np.arange(10000) + 0.5
    z_grid = 1.0 - 2.0 * i / 10000
    phi_grid = np.pi * (1.0 + np.sqrt(5.0)) * i
    r_grid = np.sqrt(1.0 - z_grid * z_grid)
    grid_points = np.column_stack([r_grid * np.cos(phi_grid), r_grid * np.sin(phi_grid), z_grid])
    
    # Generate multiple starting points
    # 1. Fibonacci Spiral
    i_spiral = np.arange(N) + 0.5
    z_spiral = 1.0 - 2.0 * i_spiral / N
    phi_spiral = np.pi * (1.0 + np.sqrt(5.0)) * i_spiral
    r_spiral = np.sqrt(1.0 - z_spiral * z_spiral)
    spiral_points = np.column_stack([r_spiral * np.cos(phi_spiral), r_spiral * np.sin(phi_spiral), z_spiral])
    
    # 2. Random Uniform
    rng = np.random.default_rng(42)
    random_points = rng.normal(size=(N, 3))
    random_points /= np.linalg.norm(random_points, axis=1, keepdims=True)
    
    starts = [spiral_points, random_points]
    
    # Add a few more random restarts if time permits
    for i in range(5):
        if time.time() - start_time > total_time_limit - 30.0:
            break
        rp = rng.normal(size=(N, 3))
        rp /= np.linalg.norm(rp, axis=1, keepdims=True)
        starts.append(rp)
        
    for idx, start_pts in enumerate(starts):
        if time.time() - start_time > total_time_limit - 20.0:
            break
            
        # Convert to spherical
        start_coords = _cartesian_to_spherical(start_pts)
        
        # Optimize
        remaining = total_time_limit - (time.time() - start_time)
        n_left = len(starts) - idx
        time_per_opt = max(5.0, (remaining - 15.0) / n_left)
        
        opt_points = _optimize_single(start_coords, time_limit=time_per_opt, grid_points=grid_points)
        
        # Normalize to ensure they are on the sphere
        opt_points /= np.linalg.norm(opt_points, axis=1, keepdims=True)
        
        # Evaluate covering radius using exact method
        score = _exact_covering_radius(opt_points)
        
        if score < best_score:
            best_score = score
            best_points = opt_points.copy()
        
        # Also evaluate with grid-based method for comparison
        grid_score = _covering_radius(opt_points, n_grid=10000)
        
        # Check if we found something very good
        if best_score < 0.197:  # ~11.3 degrees
            pass

    # Fallback if no optimization succeeded
    if best_points is None:
        best_points = spiral_points
        best_points /= np.linalg.norm(best_points, axis=1, keepdims=True)
        
    return best_points
