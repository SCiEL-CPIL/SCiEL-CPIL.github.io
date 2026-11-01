import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 130
_EPS = 1e-12

def _initial_fib():
    """Generate initial points using Fibonacci spiral."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, None))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_init(rng):
    """Generate random initial points."""
    pts = rng.standard_normal((N, 3))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts

def _get_grid():
    """
    Generate a high-resolution spherical grid for the surrogate objective.
    We use a Fibonacci lattice with ~20,000 points to ensure the grid spacing
    is small enough to approximate the true covering radius well.
    """
    M = 20000
    i = np.arange(M) + 0.5
    z = 1.0 - 2.0 * i / M
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, None))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _surrogate_objective_and_grad(x, grid):
    """
    Compute the sum of squared angular distances from grid points to nearest 
    configuration points, and its gradient with respect to configuration points.
    """
    pts = x.reshape(N, 3)
    # Normalize points to ensure they stay on the sphere during optimization steps
    # though L-BFGS-B might step off, we re-normalize in the objective calculation
    # for the distance calculation, but the gradient is with respect to the 
    # un-normalized x? No, better to optimize on the sphere directly or 
    # use a projection. 
    # Actually, standard L-BFGS-B works in R^3. We should project back or 
    # define the objective such that it's flat off the sphere? 
    # A common trick: optimize the vector x, but in the objective, normalize it.
    # The gradient of ||p|| is 0 if we treat the normalized p as the variable? 
    # No, d/dx (x/||x||) is non-trivial.
    
    # Alternative: Use the fact that for small steps, moving on the sphere 
    # is equivalent to moving in the tangent space.
    # However, implementing L-BFGS-B on the sphere requires a manifold library 
    # or careful retraction. 
    # Let's stick to R^3 optimization but penalize deviation from the sphere 
    # or just normalize inside and accept that the gradient is approximate 
    # near the sphere. 
    # Better: The gradient of the distance squared w.r.t the point p on the sphere 
    # is -2 * (grid - p) projected onto the tangent space at p.
    
    # Let's compute distances
    # dist^2 = 2 - 2 * (grid @ pts.T)
    # We want min sum_{g in grid} min_{p in pts} ||g - p||^2
    
    # Find nearest neighbor for each grid point
    # dot products
    dots = grid @ pts.T # (M, N)
    # Clip to [-1, 1]
    dots = np.clip(dots, -1.0, 1.0)
    
    # Nearest neighbor index
    nn_idx = np.argmax(dots, axis=1)
    
    # The squared angular distance is 2 * (1 - cos(theta)) = 2 - 2*dots
    # We want to minimize sum of these.
    # Objective: sum(2 - 2*dots[grid, nn_idx])
    
    # Gradient w.r.t pts[j]:
    # If grid point i is assigned to j, the term is -2 * dot(i, j).
    # d/dp_j (-2 * g_i^T p_j) = -2 * g_i.
    # But p_j is constrained to the sphere. The gradient on the sphere is the 
    # projection of the Euclidean gradient onto the tangent space.
    # Tangent space at p_j is orthogonal to p_j.
    # grad_sphere = grad_euclid - (grad_euclid . p_j) * p_j
    
    # Accumulate Euclidean gradients
    grad_euclid = np.zeros_like(pts)
    # For each grid point i, it contributes -2 * g_i to grad_euclid of its nearest neighbor nn_idx[i]
    # We can use np.add.at for this
    # Note: The factor 2 is constant, we can ignore it for optimization direction
    # Let's use just the vector g_i. Moving p towards g reduces the distance.
    # So gradient of distance squared is -2(g-p). Direction of steepest descent is g-p.
    # So we want to move p towards g.
    
    # Sum of g_i for all i assigned to j
    # grad_euclid[j] = sum_{i: nn(i)=j} g_i
    np.add.at(grad_euclid, nn_idx, grid)
    
    # Project onto tangent space
    # grad_tangent = grad_euclid - (grad_euclid . p) * p
    # p is normalized
    p_norms = np.linalg.norm(pts, axis=1, keepdims=True)
    p_norms = np.clip(p_norms, _EPS, None)
    p_unit = pts / p_norms
    
    dot_grad_p = np.sum(grad_euclid * p_unit, axis=1, keepdims=True)
    grad_tangent = grad_euclid - dot_grad_p * p_unit
    
    # Objective value
    min_dots = dots[np.arange(len(grid)), nn_idx]
    obj = np.sum(2.0 - 2.0 * min_dots)
    
    # Flatten gradient
    grad_flat = grad_tangent.flatten()
    
    return obj, grad_flat

def _hull_radius(pts):
    """Compute the exact covering radius using the convex hull."""
    try:
        hull = ConvexHull(pts)
        faces = hull.simplices
        A = pts[faces[:, 0]]
        B = pts[faces[:, 1]]
        C = pts[faces[:, 2]]
        
        AB = B - A
        AC = C - A
        cross = np.cross(AB, AC)
        area = 0.5 * np.linalg.norm(cross, axis=1)
        
        valid = area > _EPS
        if not np.any(valid):
            return np.pi
            
        normals = cross[valid]
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        
        centroids = (A[valid] + B[valid] + C[valid]) / 3.0
        signs = np.sum(normals * centroids, axis=1, keepdims=True)
        normals = normals * np.sign(signs)
        normals = normals / (norms + _EPS)
        
        cos_theta = np.sum(normals * A[valid], axis=1)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        radii = np.arccos(cos_theta)
        
        return np.max(radii)
    except Exception:
        return np.pi

def _lloyd_relax(pts, iterations=20, deadline=None):
    """
    Perform Lloyd's algorithm (Voronoi relaxation) on the sphere.
    Moves each point to the centroid of its Voronoi cell.
    """
    pts = pts.copy()
    for i in range(iterations):
        if deadline and time.time() > deadline:
            break
        try:
            hull = ConvexHull(pts)
            # Get Voronoi cells
            # scipy.spatial.Voronoi is not ideal for spherical.
            # We can use the dual of the convex hull.
            # Each vertex v is associated with the facet opposite to it? No.
            # In the convex hull, each vertex is part of several facets.
            # The Voronoi cell of a vertex on the sphere is the region of the sphere 
            # closer to it than to any other.
            # This is equivalent to the intersection of hemispheres.
            
            # A simpler approximation for Lloyd's on the sphere:
            # Move each point towards the centroid of its nearest neighbors?
            # Or use the fact that the optimal covering is related to the Delaunay triangulation.
            # Let's use a simple force-directed approach: repel close points, attract to grid centers?
            
            # Actually, a robust Lloyd's for sphere covering:
            # 1. Compute Delaunay triangulation (dual to Hull).
            # 2. For each vertex, find its adjacent facets.
            # 3. The centroid of the Voronoi cell is the point on the sphere equidistant 
            #    to the vertices of the adjacent facets? No.
            
            # Let's use a gradient step towards the "center of mass" of the 
            # points in the Voronoi cell. Since we don't have exact cells, 
            # we can approximate by moving towards the centroid of the 
            # nearest K points?
            
            # Better: Use the normal vectors of the facets incident to the vertex.
            # The vertex should move towards the average of the normals of the 
            # facets it belongs to?
            
            # Let's stick to the subgradient of the max radius, but apply it 
            # to ALL vertices, weighted by their contribution to the max radius.
            
            # For now, let's just do a few steps of the subgradient descent 
            # but on all points, not just the worst facet.
            
            radii, poles, faces = _hull_radii_and_poles(pts)
            if len(radii) == 0:
                break
                
            curr_r = np.max(radii)
            
            # For each vertex, calculate the "pressure" from its incident facets
            # We want to move vertices that are in large-radius facets towards the poles.
            
            # Map facet to vertices
            # We can accumulate the pole direction for each vertex, weighted by 
            # (radius - average_radius) or just radius.
            
            target_dirs = np.zeros_like(pts)
            weights = np.zeros(N)
            
            for idx in range(len(radii)):
                r = radii[idx]
                pole = poles[idx]
                verts = faces[idx]
                # Distribute the force equally among the 3 vertices
                for v in verts:
                    target_dirs[v] += pole * r
                    weights[v] += r
            
            # Normalize and step
            mask = weights > 0
            if np.any(mask):
                target_dirs[mask] /= (weights[mask, None] + _EPS)
                # Ensure target_dirs are unit vectors
                norms = np.linalg.norm(target_dirs, axis=1, keepdims=True)
                norms = np.clip(norms, _EPS, None)
                target_dirs = target_dirs / norms
                
                step = 0.02
                # Move towards target
                new_pts = pts.copy()
                new_pts[mask] = pts[mask] + step * target_dirs[mask]
                new_pts[mask] /= (np.linalg.norm(new_pts[mask], axis=1, keepdims=True) + _EPS)
                
                # Check if it improved
                new_r = _hull_radius(new_pts)
                if new_r < curr_r:
                    pts = new_pts
                else:
                    # Try smaller step
                    step *= 0.5
                    new_pts = pts.copy()
                    new_pts[mask] = pts[mask] + step * target_dirs[mask]
                    new_pts[mask] /= (np.linalg.norm(new_pts[mask], axis=1, keepdims=True) + _EPS)
                    new_r = _hull_radius(new_pts)
                    if new_r < curr_r:
                        pts = new_pts
                    else:
                        # If still not improving, break or try random perturb
                        pass
                        
        except Exception:
            break
            
    return pts

def _hull_radii_and_poles(pts):
    """
    Compute the covering radius and the poles (circumcenters) of all facets.
    Returns:
        radii: array of angular radii for each valid facet
        poles: array of unit vectors pointing to the circumcenters
        faces: the simplices of the convex hull
    """
    try:
        hull = ConvexHull(pts)
        faces = hull.simplices
        M = len(faces)
        
        A = pts[faces[:, 0]]
        B = pts[faces[:, 1]]
        C = pts[faces[:, 2]]
        
        AB = B - A
        AC = C - A
        cross = np.cross(AB, AC)
        area = 0.5 * np.linalg.norm(cross, axis=1)
        
        # Filter out degenerate facets
        valid = area > _EPS
        if not np.any(valid):
            return np.array([np.pi]), np.array([[1.0, 0.0, 0.0]]), faces

        normals = cross[valid]
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        
        # Ensure normals point outward
        centroids = (A[valid] + B[valid] + C[valid]) / 3.0
        signs = np.sum(normals * centroids, axis=1, keepdims=True)
        normals = normals * np.sign(signs)
        normals = normals / (norms + _EPS)
        
        cos_theta = np.sum(normals * A[valid], axis=1)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        radii = np.arccos(cos_theta)
        
        return radii, normals, faces[valid]

    except Exception:
        return np.array([np.pi]), np.array([[1.0, 0.0, 0.0]]), np.empty((0, 3), dtype=int)

def _multi_restart_search(deadline):
    """
    Run multiple restarts of smooth optimization + Lloyd's relaxation.
    """
    rng = np.random.default_rng(42)
    best_pts = _initial_fib()
    best_r = _hull_radius(best_pts)
    
    grid = _get_grid()
    
    n_restarts = 5
    time_per_restart = (deadline - 10.0) / n_restarts
    
    for restart in range(n_restarts):
        if time.time() > deadline - 10.0:
            break
            
        if restart == 0:
            pts = _initial_fib()
        else:
            pts = _random_init(rng)
        
        # Phase 1: Smooth optimization using L-BFGS-B
        # We optimize in R^3, but the objective handles normalization.
        # We need to be careful with the gradient.
        
        x0 = pts.flatten()
        
        # To make L-BFGS-B work well, we might need to scale the problem.
        # The objective is sum of squared distances, which is smooth.
        
        try:
            res = minimize(
                _surrogate_objective_and_grad,
                x0,
                args=(grid,),
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            # Normalize the result
            pts_opt = res.x.reshape(N, 3)
            pts_opt /= (np.linalg.norm(pts_opt, axis=1, keepdims=True) + _EPS)
        except Exception:
            pts_opt = pts
            
        # Phase 2: Lloyd's relaxation / Local polish
        local_deadline = time.time() + time_per_restart * 0.5
        pts_polished = _lloyd_relax(pts_opt, iterations=20, deadline=local_deadline)
        
        # Evaluate
        curr_r = _hull_radius(pts_polished)
        
        if curr_r < best_r:
            best_r = curr_r
            best_pts = pts_polished.copy()
            
    # Final polish with the best found
    if time.time() < deadline - 5.0:
        best_pts = _lloyd_relax(best_pts, iterations=50, deadline=deadline)
        best_r = _hull_radius(best_pts)
        
    return best_pts

def construct():
    deadline = time.time() + 145.0  # Safety margin
    best_pts = _multi_restart_search(deadline)
    
    # Final normalization and validity check
    best_pts = np.array(best_pts, dtype=float)
    best_pts /= (np.linalg.norm(best_pts, axis=1, keepdims=True) + _EPS)
    
    if not np.all(np.isfinite(best_pts)) or best_pts.shape != (N, 3):
        return _initial_fib()
        
    return best_pts
