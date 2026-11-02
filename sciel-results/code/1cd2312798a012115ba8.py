import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time
import warnings

N = 130

# EVOLVE-BLOCK-START

def _normalize(v):
    """Normalize vectors to unit length."""
    norms = np.linalg.norm(v, axis=1)
    norms[norms == 0] = 1.0
    return v / norms[:, np.newaxis]

def _covering_radius_from_hull(points):
    """
    Compute the exact covering radius (max circumradius of convex hull facets) in radians.
    """
    if len(points) < 3:
        return 0.0
    
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi
        
    max_radius = 0.0
    for simplex in hull.simplices:
        v1, v2, v3 = points[simplex]
        
        # Compute circumcenter of the triangle on the sphere
        n = np.cross(v2 - v1, v3 - v1)
        norm_n = np.linalg.norm(n)
        if norm_n < 1e-12:
            continue # Degenerate triangle
            
        c = n / norm_n
        
        # Ensure c points outward (towards the centroid of the triangle)
        centroid = (v1 + v2 + v3) / 3.0
        if np.dot(c, centroid) < 0:
            c = -c
            
        # The angular distance from c to any vertex is the circumradius
        cos_angle = np.dot(c, v1)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        
        if angle > max_radius:
            max_radius = angle
            
    return max_radius

def _soft_covering_radius_and_grad(points, grid_points, sigma=0.05):
    """
    Vectorized soft covering radius and its gradient.
    Uses soft-min over distances from grid points to config points,
    and soft-max over the resulting nearest-distances.
    """
    # Compute angular distances: (M, N)
    dots = grid_points @ points.T
    dots = np.clip(dots, -1.0, 1.0)
    angles = np.arccos(dots)  # (M, N)
    
    # Soft-min: for each grid point, compute weighted average of angles
    # w_j = exp(-angle_j / sigma) / sum_k exp(-angle_k / sigma)
    neg_scaled = -angles / sigma  # (M, N)
    max_neg = np.max(neg_scaled, axis=1, keepdims=True)  # (M, 1)
    exp_neg = np.exp(neg_scaled - max_neg)  # (M, N)
    weights = exp_neg / np.sum(exp_neg, axis=1, keepdims=True)  # (M, N)
    
    # Soft-min value: weighted sum of angles
    soft_min_vals = np.sum(weights * angles, axis=1)  # (M,)
    
    # Soft-max of soft-min values (to get covering radius approximation)
    scaled_max = soft_min_vals / sigma
    max_scaled = np.max(scaled_max)
    exp_max = np.exp(scaled_max - max_scaled)
    max_weights = exp_max / np.sum(exp_max)  # (M,)
    soft_max_val = sigma * (max_scaled + np.log(np.sum(exp_max)))
    
    # Gradient computation
    # d(soft_min_i)/d(points_j) = sum_k weights_ik * d(angle_ik)/d(points_j)
    # d(angle_ik)/d(points_j) = 0 if j != k, else -g_i / sqrt(1 - dot^2) projected to tangent
    
    # For each grid point i and config point j:
    # contribution = max_weights[i] * weights[i,j] * (-g_i / sqrt(1 - dots[i,j]^2)) * tangent_projection
    
    # denom: (M, N)
    denom = np.sqrt(np.maximum(1.0 - dots * dots, 1e-18))  # (M, N)
    
    # coeff[i,j] = max_weights[i] * weights[i,j]
    coeff = max_weights[:, np.newaxis] * weights  # (M, N)
    
    # alpha[i,j] = coeff[i,j] / denom[i,j]
    alpha = coeff / denom  # (M, N)
    
    # Term1: -sum_i alpha[i,j] * g_i
    # This is: -(alpha.T @ grid_points)
    term1 = -(alpha.T @ grid_points)  # (N, 3)
    
    # Term2: sum_i alpha[i,j] * dots[i,j] * p_j
    beta = alpha * dots  # (M, N)
    beta_sum = np.sum(beta, axis=0)  # (N,)
    term2 = beta_sum[:, np.newaxis] * points  # (N, 3)
    
    grad = term1 + term2  # (N, 3)
    
    return soft_max_val, grad

def _generate_grid(n_theta=60, n_phi=120):
    """Generate a grid of points on the sphere for soft radius calculation."""
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing='ij')
    
    x = np.sin(theta_grid) * np.cos(phi_grid)
    y = np.sin(theta_grid) * np.sin(phi_grid)
    z = np.cos(theta_grid)
    
    grid = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    return grid

def _lloyd_step(points, grid_points, n_iters=3):
    """
    Perform Lloyd's algorithm iterations using grid-based Voronoi.
    Uses vectorized operations for speed.
    """
    for _ in range(n_iters):
        dots = grid_points @ points.T
        angles = np.arccos(np.clip(dots, -1.0, 1.0))
        nearest = np.argmin(angles, axis=1)
        
        new_points = np.zeros_like(points)
        
        # Vectorized: sum grid points per cluster, then divide by count
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

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin for 150s limit
    
    # 1. Initial configuration: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    points = _normalize(points)
    
    # 2. Gradient-based optimization using soft covering radius
    # Use a moderate resolution grid to balance speed and accuracy
    grid_points = _generate_grid(40, 80)
    
    def objective_and_grad(x):
        pts = _normalize(x.reshape(-1, 3))
        val, grad = _soft_covering_radius_and_grad(pts, grid_points, sigma=0.05)
        return val, grad.ravel()
    
    try:
        res = minimize(
            objective_and_grad,
            points.ravel(),
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6}
        )
        points = _normalize(res.x.reshape(-1, 3))
    except Exception:
        pass # Keep initial points if optimization fails
    
    # 3. Lloyd's algorithm to improve uniformity (CVT-like)
    if time.time() < deadline - 30:
        points = _lloyd_step(points, grid_points, n_iters=5)
    
    # 4. Exact geometric refinement
    # Iteratively find the worst facet and move its vertices towards the circumcenter
    best_points = points.copy()
    best_radius = _covering_radius_from_hull(best_points)
    
    # Adaptive step size
    alpha = 0.1
    max_iter = 5000
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
            
        try:
            hull = ConvexHull(points)
        except Exception:
            break
            
        max_radius = 0.0
        worst_simplex = None
        worst_center = None
        
        for simplex in hull.simplices:
            v1, v2, v3 = points[simplex]
            n = np.cross(v2 - v1, v3 - v1)
            norm_n = np.linalg.norm(n)
            if norm_n < 1e-12:
                continue
            c = n / norm_n
            
            # Ensure orientation
            centroid = (v1 + v2 + v3) / 3.0
            if np.dot(c, centroid) < 0:
                c = -c
                
            cos_angle = np.dot(c, v1)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle)
            
            if angle > max_radius:
                max_radius = angle
                worst_simplex = simplex
                worst_center = c
                
        if worst_simplex is None:
            break
            
        # Update best if improved
        if max_radius < best_radius:
            best_radius = max_radius
            best_points = points.copy()
            
        # Check convergence
        if max_radius < 11.3165625 * np.pi / 180.0 + 1e-6:
            break
            
        # Move vertices of the worst simplex towards the center
        # We move them slightly to reduce the circumradius
        for idx in worst_simplex:
            points[idx] = (1.0 - alpha) * points[idx] + alpha * worst_center
            
        # Normalize all points back to the sphere
        points = _normalize(points)
        
        # Check if we made progress
        new_radius = _covering_radius_from_hull(points)
        if new_radius > max_radius:
            # Revert and reduce step size
            points = best_points.copy()
            alpha = max(alpha * 0.5, 0.01)
        else:
            # Increase step size slightly if we're making good progress
            if new_radius < max_radius * 0.99:
                alpha = min(alpha * 1.1, 0.2)
        
        # Occasionally add noise to escape local minima
        if it % 50 == 0 and it > 0:
            noise = np.random.normal(0, 0.01, points.shape)
            points = _normalize(points + noise)

    # Final check: if best_points is better than current points, use best_points
    # (The loop updates best_points only on improvement, but the last step might have degraded it)
    final_radius = _covering_radius_from_hull(best_points)
    current_radius = _covering_radius_from_hull(points)
    
    if final_radius < current_radius:
        points = best_points
    else:
        points = points

    # Final normalization and validation
    points = _normalize(points)
    
    # Check for coincident points (should not happen with normalization and noise, but just in case)
    # If any two points are too close, we might have a degenerate hull
    # For this problem, distinct points are required.
    
    return points

# EVOLVE-BLOCK-END
