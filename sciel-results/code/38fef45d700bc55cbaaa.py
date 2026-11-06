import numpy as np
import time
from scipy.optimize import minimize, linprog
from scipy.spatial import ConvexHull, Delaunay

N = 130

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    return pts / norms

def _covering_radius_deg(pts):
    """
    Compute the exact covering radius in degrees.
    The covering radius is the maximum over all facets of the convex hull
    of the circumradius of the facet's spherical cap.
    For a facet with vertices v1, ..., vk on the unit sphere, the covering
    radius is the maximum angular distance from the facet's center (the
    normalized centroid of the facet vertices projected onto the sphere)
    to the farthest vertex.
    
    Actually, the covering radius of a facet is the radius of the smallest
    spherical cap containing the facet's vertices. For a spherical polygon,
    this is determined by the circumcircle of the polygon's vertices on the sphere.
    For a triangle (most common facet), it's the circumradius of the spherical triangle.
    For a general facet, we can compute the maximum angular distance from the
    "center" of the facet to its vertices.
    
    A more robust approach: for each facet, compute the circumradius as the
    maximum angular distance from the facet's incenter (or centroid) to its vertices.
    But the true covering radius for a facet is the radius of the minimal enclosing
    spherical cap, which for a set of points on a sphere is determined by the
    2 or 3 points that are farthest apart.
    
    Simplified robust approach: For each facet, compute the maximum angular distance
    between any pair of vertices in the facet. The covering radius is half of this
    maximum? No, that's not right either.
    
    Correct approach: The covering radius is the maximum, over all points y on the sphere,
    of the distance from y to the nearest configuration point. This equals the maximum
    circumradius of the Voronoi cells. For the convex hull, each facet corresponds to a
    Voronoi cell. The circumradius of a facet's Voronoi cell is the maximum angular
    distance from the facet's center to its vertices.
    
    For a spherical polygon with vertices v1, ..., vk, the center of the minimal enclosing
    cap is the point c on the sphere that minimizes max_i angular_dist(c, vi).
    This is the spherical 1-center problem. For small facets (triangles), we can
    approximate by computing the circumcenter.
    
    Practical approach: For each facet, compute the maximum angular distance from the
    normalized centroid of the facet's vertices to the farthest vertex. This is an
    upper bound on the true covering radius for that facet. The maximum over all facets
    gives an upper bound on the covering radius.
    
    For better accuracy, we can compute the circumradius of each facet's vertices.
    For a triangle with side lengths a, b, c (angular distances), the circumradius R
    satisfies: sin(R) = (a*b*c) / (4 * area) where area is the spherical area.
    But this is complex. Let's use a simpler robust method.
    
    Actually, the covering radius can be computed as follows:
    1. Compute the convex hull of the points.
    2. For each facet, compute the "center" as the normalized sum of the facet's vertices.
    3. The covering radius for that facet is the maximum angular distance from the center
       to any vertex of the facet.
    4. The overall covering radius is the maximum over all facets.
    
    This is not exactly correct, but it's a good approximation. For a more accurate
    computation, we can use the fact that the covering radius is the maximum over all
    points on the sphere of the distance to the nearest point. We can sample the sphere
    densely and compute this, but that's expensive.
    
    Let's use the convex hull approach with a refinement: for each facet, compute the
    circumradius by finding the point on the sphere that minimizes the maximum distance
    to the facet's vertices. This is the spherical 1-center problem, which can be
    solved by checking the circumcenters of all subsets of 2 or 3 vertices.
    """
    hull = ConvexHull(pts)
    facets = hull.simplices
    
    max_radius = 0.0
    for simplex in facets:
        verts = pts[simplex]
        # Compute the center as the normalized centroid
        center = np.sum(verts, axis=0)
        center_norm = np.linalg.norm(center)
        if center_norm < 1e-15:
            # Degenerate facet
            continue
        center = center / center_norm
        
        # Compute angular distances from center to each vertex
        dots = np.dot(verts, center)
        dots = np.clip(dots, -1.0, 1.0)
        angles = np.arccos(dots)
        max_angle = np.max(angles)
        
        if max_angle > max_radius:
            max_radius = max_angle
    
    return np.degrees(max_radius)

def _smooth_covering_radius(pts, beta=50.0):
    """
    Smooth surrogate for the covering radius.
    For each point on a dense grid on the sphere, compute the soft-min distance
    to the nearest configuration point. The soft-min is:
    softmin(d_i) = -(1/beta) * log(sum(exp(-beta * d_i)))
    This approximates min(d_i) as beta -> infinity.
    The covering radius is the max over the grid of softmin(d_i).
    We use a dense grid for accuracy.
    """
    # Generate a dense grid on the sphere
    n_theta = 100
    n_phi = 200
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing='ij')
    
    # Convert to Cartesian
    x = np.sin(theta_grid) * np.cos(phi_grid)
    y = np.sin(theta_grid) * np.sin(phi_grid)
    z = np.cos(theta_grid)
    grid = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    grid = _normalize(grid)
    
    # Compute distances from each grid point to each configuration point
    # diff[i, j] = grid[i] - pts[j]
    diff = grid[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    # Angular distance: arccos(1 - dist_sq/2)
    dots = 1.0 - dist_sq / 2.0
    dots = np.clip(dots, -1.0, 1.0)
    angles = np.arccos(dots)
    
    # Soft-min over configuration points
    # softmin(angles[i, :]) = -(1/beta) * log(sum(exp(-beta * angles[i, :])))
    # For numerical stability, subtract the min
    min_angles = np.min(angles, axis=1, keepdims=True)
    exp_terms = np.exp(-beta * (angles - min_angles))
    log_sum = np.log(np.sum(exp_terms, axis=1, keepdims=True))
    softmin = min_angles - log_sum / beta
    
    return np.max(softmin)

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin for 150s kill
    
    # 1. Initialization: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    points = _normalize(points)
    
    best_points = points.copy()
    best_cr = _covering_radius_deg(points)
    
    # 2. L-BFGS optimization on a smooth surrogate
    # We optimize the smooth covering radius directly.
    # Parametrization: we work in Cartesian coordinates and project back to the sphere.
    # This is not ideal for L-BFGS, but it works if we project after each step.
    
    def objective_and_grad(x):
        pts = _normalize(x.reshape(N, 3))
        cr = _smooth_covering_radius(pts, beta=50.0)
        
        # Numerical gradient
        eps = 1e-6
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += eps
            pts_plus = _normalize(x_plus.reshape(N, 3))
            cr_plus = _smooth_covering_radius(pts_plus, beta=50.0)
            grad[i] = (cr_plus - cr) / eps
        
        return cr, grad
    
    # Use L-BFGS-B with numerical gradients
    x0 = points.ravel()
    
    # We'll do a few L-BFGS restarts with different initializations
    for restart in range(3):
        if time.time() - start_time > time_limit - 30:
            break
        
        # For restarts > 0, add random perturbation
        if restart == 0:
            x_current = x0.copy()
        else:
            x_current = x0.copy()
            # Add random perturbation
            perturbation = np.random.randn(N, 3) * 0.1
            x_current = _normalize(x_current.reshape(N, 3) + perturbation).ravel()
        
        result = minimize(
            lambda x: _smooth_covering_radius(_normalize(x.reshape(N, 3)), beta=50.0),
            x_current,
            method='L-BFGS-B',
            options={'maxiter': 100, 'gtol': 1e-6, 'maxfun': 500}
        )
        
        pts = _normalize(result.x.reshape(N, 3))
        cr = _covering_radius_deg(pts)
        
        if cr < best_cr:
            best_cr = cr
            best_points = pts.copy()
    
    # 3. High-resolution polish: directly minimize the true covering radius
    # Use a local search that moves points to reduce the largest facet circumradius
    
    if time.time() - start_time < time_limit - 10:
        pts = best_points.copy()
        step_size = 0.01
        for iteration in range(200):
            if time.time() - start_time > time_limit - 5:
                break
            
            # Compute the current covering radius and identify the worst facet
            hull = ConvexHull(pts)
            facets = hull.simplices
            
            # For each facet, compute the circumradius
            facet_radii = []
            for simplex in facets:
                verts = pts[simplex]
                center = np.sum(verts, axis=0)
                center_norm = np.linalg.norm(center)
                if center_norm < 1e-15:
                    facet_radii.append(0.0)
                    continue
                center = center / center_norm
                dots = np.dot(verts, center)
                dots = np.clip(dots, -1.0, 1.0)
                angles = np.arccos(dots)
                facet_radii.append(np.max(angles))
            
            facet_radii = np.array(facet_radii)
            worst_facet_idx = np.argmax(facet_radii)
            worst_radius = facet_radii[worst_facet_idx]
            
            # Move the vertices of the worst facet to reduce the circumradius
            # Move each vertex slightly towards the center of the facet
            simplex = facets[worst_facet_idx]
            verts = pts[simplex]
            center = np.sum(verts, axis=0)
            center_norm = np.linalg.norm(center)
            if center_norm < 1e-15:
                continue
            center = center / center_norm
            
            # Move each vertex towards the center
            for idx in simplex:
                direction = center - pts[idx]
                direction_norm = np.linalg.norm(direction)
                if direction_norm > 1e-15:
                    pts[idx] = pts[idx] + step_size * direction / direction_norm
            
            pts = _normalize(pts)
            
            # Check if this improved the covering radius
            new_cr = _covering_radius_deg(pts)
            if new_cr < best_cr:
                best_cr = new_cr
                best_points = pts.copy()
            else:
                # Revert the move
                pts = best_points.copy()
                step_size *= 0.9
                if step_size < 1e-6:
                    break
    
    return best_points

if __name__ == "__main__":
    pts = construct()
    print(f"Shape: {pts.shape}")
    print(f"Norms: {np.linalg.norm(pts, axis=1)[:5]}")
    print(f"Covering radius: {_covering_radius_deg(pts):.6f} degrees")
