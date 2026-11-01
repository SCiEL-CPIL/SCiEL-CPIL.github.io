"""Combined program: Lloyd's Algorithm + Soft Gradient Optimization for Sphere Covering."""
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import sys

N = 130

# EVOLVE-BLOCK-START

def _compute_exact_covering_radius(points):
    """
    Compute the exact covering radius using the convex hull.
    The covering radius is the maximum circumradius of the spherical triangles (facets).
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi / 2.0 # Fallback if hull fails, should be rare

    max_radius = 0.0
    for simplex in hull.simplices:
        verts = points[simplex]
        a, b, c = verts[0], verts[1], verts[2]
        
        # Compute normal vector to the facet
        # The circumcenter of a spherical triangle lies on the normal to the plane defined by the vertices
        # specifically, the pole of the great circle passing through the vertices? No.
        # The circumcenter o is the point on the sphere equidistant from a, b, c.
        # o is parallel to (b-a) x (c-a) ?
        # Let's verify.
        # (o - a) . (b - a) = 0 => o . (b-a) = a . (b-a)
        # (o - a) . (c - a) = 0 => o . (c-a) = a . (c-a)
        # This implies o is in the intersection of two planes.
        # The vector n = (b-a) x (c-a) is perpendicular to both (b-a) and (c-a).
        # So o is parallel to n.
        
        n = np.cross(b - a, c - a)
        norm_n = np.linalg.norm(n)
        if norm_n < 1e-10:
            continue
        
        o = n / norm_n
        
        # Check orientation: o should be on the same side of the origin as the centroid of the facet
        centroid = (a + b + c) / 3.0
        if np.dot(o, centroid) < 0:
            o = -o
            
        # Calculate angular distance from o to a (or b or c)
        # cos(R) = o . a
        cos_r = np.dot(o, a)
        cos_r = np.clip(cos_r, -1.0, 1.0)
        r_val = np.arccos(cos_r)
        
        if r_val > max_radius:
            max_radius = r_val
            
    return max_radius

def _soft_covering_radius_and_grad(points, grid_points, sigma=0.05):
    """
    Compute a soft approximation of the covering radius and its gradient.
    Uses a grid of points on the sphere to approximate the max over the sphere.
    For each grid point, it finds the nearest configuration point (soft min).
    Then it takes the soft max over all grid points.
    """
    # points: (N, 3)
    # grid_points: (M, 3)
    
    # Compute dot products: (M, N)
    dots = grid_points @ points.T
    dots = np.clip(dots, -1.0, 1.0)
    angles = np.arccos(dots)  # (M, N)
    
    # Soft-min over configuration points for each grid point
    # w_j = exp(-angle_j / sigma) / sum_k exp(-angle_k / sigma)
    neg_scaled = -angles / sigma  # (M, N)
    max_neg = np.max(neg_scaled, axis=1, keepdims=True)  # (M, 1)
    exp_neg = np.exp(neg_scaled - max_neg)  # (M, N)
    weights = exp_neg / np.sum(exp_neg, axis=1, keepdims=True)  # (M, N)
    
    # Soft-min value: weighted sum of angles
    soft_min_vals = np.sum(weights * angles, axis=1)  # (M,)
    
    # Soft-max over grid points
    scaled_max = soft_min_vals / sigma
    max_scaled = np.max(scaled_max)
    exp_max = np.exp(scaled_max - max_scaled)
    max_weights = exp_max / np.sum(exp_max)  # (M,)
    soft_max_val = sigma * (max_scaled + np.log(np.sum(exp_max)))
    
    # Gradient computation
    # d(soft_min_i)/d(points_j) = sum_k weights_ik * d(angle_ik)/d(points_j)
    # d(angle_ik)/d(points_j) = 0 if j != k
    # For j == k: d(arccos(g_i . p_j))/dp_j = -g_i / sqrt(1 - (g_i . p_j)^2)
    # But we need the tangent component.
    # grad_raw = -g_i / sqrt(1 - dot^2)
    # grad_tangent = grad_raw - (grad_raw . p_j) * p_j
    # grad_raw . p_j = -dot(g_i, p_j) / sqrt(1 - dot^2) = -dots[i,j] / sqrt(1 - dot^2)
    # grad_tangent = -g_i / sqrt(1 - dot^2) + dots[i,j] / sqrt(1 - dot^2) * p_j
    
    denom = np.sqrt(np.maximum(1.0 - dots * dots, 1e-18))  # (M, N)
    
    # coeff[i,j] = max_weights[i] * weights[i,j]
    coeff = max_weights[:, np.newaxis] * weights  # (M, N)
    
    # alpha[i,j] = coeff[i,j] / denom[i,j]
    alpha = coeff / denom  # (M, N)
    
    # Term 1: sum_i alpha[i,j] * (-g_i)
    # = -(alpha.T @ grid_points)
    term1 = -(alpha.T @ grid_points)  # (N, 3)
    
    # Term 2: sum_i alpha[i,j] * dots[i,j] * p_j
    # For each j: p_j * sum_i alpha[i,j] * dots[i,j]
    beta = alpha * dots  # (M, N)
    beta_sum = np.sum(beta, axis=0)  # (N,)
    term2 = beta_sum[:, np.newaxis] * points  # (N, 3)
    
    grad = term1 + term2  # (N, 3)
    
    return soft_max_val, grad

def _lloyd_step(points, grid_points):
    """
    Perform one iteration of Lloyd's algorithm.
    Move each point to the spherical centroid of its Voronoi cell.
    """
    # Compute angular distances
    dots = grid_points @ points.T
    dots = np.clip(dots, -1.0, 1.0)
    angles = np.arccos(dots)
    nearest = np.argmin(angles, axis=1)
    
    new_points = np.zeros_like(points)
    
    # Vectorized update
    # We can use np.add.at for efficient accumulation
    # But simple loop is fine for N=130 and moderate M
    for j in range(N):
        mask = (nearest == j)
        count = np.sum(mask)
        if count > 0:
            # Spherical centroid is the normalized sum of unit vectors
            new_points[j] = np.sum(grid_points[mask], axis=0)
        else:
            new_points[j] = points[j]
        
    norms = np.linalg.norm(new_points, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    new_points = new_points / norms
    
    return new_points

def construct():
    start_time = time.time()
    deadline = 145.0  # 5s safety margin
    
    # 1. Initialization: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Create a grid of points for soft metrics and Lloyd's algorithm
    # Using a Fibonacci spiral for the grid as well, with more points
    M = 2000
    i_grid = np.arange(M) + 0.5
    z_grid = 1.0 - 2.0 * i_grid / M
    phi_grid = np.pi * (1.0 + np.sqrt(5.0)) * i_grid
    r_grid = np.sqrt(1.0 - z_grid * z_grid)
    grid_points = np.column_stack([r_grid * np.cos(phi_grid), r_grid * np.sin(phi_grid), z_grid])
    
    best_points = points.copy()
    best_radius = _compute_exact_covering_radius(points)
    
    # 2. Lloyd's Algorithm Phase
    # Run a few iterations to get a good initial distribution
    num_lloyd_iters = 20
    for _ in range(num_lloyd_iters):
        if time.time() > deadline - 30: # Reserve 30s for gradient step
            break
        points = _lloyd_step(points, grid_points)
        current_radius = _compute_exact_covering_radius(points)
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = points.copy()
            
    # 3. Gradient-Based Optimization Phase
    # Use L-BFGS-B to minimize the soft covering radius
    # We flatten the points to a 1D array for scipy.optimize
    
    def objective_and_grad(flat_points):
        pts = flat_points.reshape(N, 3)
        val, grad = _soft_covering_radius_and_grad(pts, grid_points, sigma=0.05)
        flat_grad = grad.flatten()
        return val, flat_grad
    
    # Initial point for optimization
    current_flat = points.flatten()
    
    # Optimize
    try:
        res = minimize(
            objective_and_grad,
            current_flat,
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': 50, 'gtol': 1e-8, 'maxls': 50}
        )
        
        if res.success or res.fun < best_radius:
            opt_points = res.x.reshape(N, 3)
            # Normalize
            opt_points = opt_points / np.linalg.norm(opt_points, axis=1, keepdims=True)
            
            # Check exact radius
            opt_radius = _compute_exact_covering_radius(opt_points)
            if opt_radius < best_radius:
                best_radius = opt_radius
                best_points = opt_points.copy()
                
    except Exception as e:
        pass # If optimization fails, keep previous best
        
    # 4. Final Polish: A few more Lloyd iterations on the best points found
    # This helps center points in their Voronoi cells, which often reduces max facet radius
    points = best_points.copy()
    num_polish_iters = 10
    for _ in range(num_polish_iters):
        if time.time() > deadline - 5:
            break
        points = _lloyd_step(points, grid_points)
        current_radius = _compute_exact_covering_radius(points)
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = points.copy()

    # Final validation
    # Ensure points are on the sphere
    best_points = best_points / np.linalg.norm(best_points, axis=1, keepdims=True)
    
    # Check for coincident points or invalid hulls
    try:
        _ = ConvexHull(best_points)
    except Exception:
        # If hull fails, return initial Fibonacci points
        return points

    return best_points

# EVOLVE-BLOCK-END
