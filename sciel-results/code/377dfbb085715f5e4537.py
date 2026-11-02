"""Covering the sphere with 50 equal caps - optimized solver using epigraph formulation."""
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 50

def _normalize(points):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(points, axis=1)
    # Handle any zero vectors
    norms[norms < 1e-14] = 1.0
    return points / norms[:, np.newaxis]

def _compute_facet_data(points):
    """Compute convex hull and return facet data.
    
    Returns:
        facets: array of shape (n_facets, 3) with vertex indices
        facet_centers: array of shape (n_facets, 3) with normalized facet centers
        facet_radii: array of shape (n_facets,) with angular radius of each facet
        valid: boolean indicating if hull computation succeeded
    """
    try:
        hull = ConvexHull(points)
        facets = hull.simplices  # (n_facets, 3)
        n_facets = len(facets)
        
        # Compute facet centers and radii
        facet_centers = np.zeros((n_facets, 3))
        facet_radii = np.zeros(n_facets)
        
        for i, (a, b, c) in enumerate(facets):
            pa = points[a]
            pb = points[b]
            pc = points[c]
            
            # Facet center is the normalized cross product of two edge vectors
            # Or more robustly: the normalized normal to the plane
            edge1 = pb - pa
            edge2 = pc - pa
            normal = np.cross(edge1, edge2)
            nnorm = np.linalg.norm(normal)
            if nnorm < 1e-14:
                facet_centers[i] = pa  # Degenerate, use a vertex
                facet_radii[i] = 0.0
                continue
            normal /= nnorm
            
            # The circumcenter of the spherical triangle is in the direction of the normal
            # But we need to check it's on the correct side
            # The center should be such that it's "above" the plane
            # Check: dot(center, pa) should be close to cos(radius)
            
            # For a spherical triangle, the circumcenter direction is the normalized
            # vector that is equidistant from all three vertices
            # This is the direction of the normal to the plane containing the three points
            # but we need to make sure it points outward from the origin
            
            # Actually, for points on the unit sphere, the circumcenter of the spherical
            # triangle is the point on the sphere in the direction of the plane normal
            # (choosing the sign so that it's on the same side as the points)
            
            # Check which direction: dot with midpoint of edge
            mid = (pa + pb) / 2.0
            if np.dot(normal, mid) < 0:
                normal = -normal
            
            facet_centers[i] = normal
            
            # Compute angular radius: arccos(dot(center, pa))
            dot_val = np.dot(normal, pa)
            dot_val = np.clip(dot_val, -1.0, 1.0)
            facet_radii[i] = np.arccos(dot_val)
        
        return facets, facet_centers, facet_radii, True
        
    except Exception:
        return None, None, None, False

def _compute_covering_radius(points):
    """Compute the covering radius in degrees."""
    points = _normalize(points)
    facets, facet_centers, facet_radii, valid = _compute_facet_data(points)
    
    if not valid or len(facet_radii) == 0:
        # Fallback: dense sampling
        n_samples = 100000
        i = np.arange(n_samples) + 0.5
        z = 1.0 - 2.0 * i / n_samples
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.maximum(0, 1.0 - z * z))
        sample_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        
        dots = sample_points @ points.T
        min_dots = np.max(dots, axis=1)
        angles = np.arccos(np.clip(min_dots, -1.0, 1.0))
        return np.degrees(np.max(angles))
    
    return np.degrees(np.max(facet_radii))

def _soft_max_radius_and_grad(points, temperature=0.01):
    """Compute a smooth approximation of the max facet radius and its gradient.
    
    Uses log-sum-exp for smooth max, and computes gradient via finite differences
    of the smooth objective.
    
    Returns:
        value: smooth max radius (in radians)
        grad: gradient w.r.t. points (N, 3)
    """
    points = _normalize(points)
    facets, facet_centers, facet_radii, valid = _compute_facet_data(points)
    
    if not valid or len(facet_radii) == 0:
        return np.pi, np.zeros_like(points)
    
    # Log-sum-exp for smooth max
    max_r = np.max(facet_radii)
    shifted = facet_radii - max_r
    weights = np.exp(shifted / temperature)
    weights /= np.sum(weights)
    smooth_max = max_r + temperature * np.log(np.sum(weights))
    
    # Compute gradient using finite differences
    grad = np.zeros_like(points)
    eps = 1e-6
    
    for i in range(N):
        for j in range(3):
            # Tangent direction
            e = np.zeros(3)
            e[j] = 1.0
            tangent = e - np.dot(e, points[i]) * points[i]
            tnorm = np.linalg.norm(tangent)
            if tnorm < 1e-10:
                continue
            tangent /= tnorm
            
            pts_plus = points.copy()
            pts_plus[i] = points[i] + eps * tangent
            pts_plus[i] = pts_plus[i] / np.linalg.norm(pts_plus[i])
            
            _, _, radii_plus, valid_plus = _compute_facet_data(pts_plus)
            if valid_plus and len(radii_plus) > 0:
                max_plus = np.max(radii_plus)
                shifted_plus = radii_plus - max_plus
                weights_plus = np.exp(shifted_plus / temperature)
                weights_plus /= np.sum(weights_plus)
                smooth_plus = max_plus + temperature * np.log(np.sum(weights_plus))
            else:
                smooth_plus = np.pi
            
            pts_minus = points.copy()
            pts_minus[i] = points[i] - eps * tangent
            pts_minus[i] = pts_minus[i] / np.linalg.norm(pts_minus[i])
            
            _, _, radii_minus, valid_minus = _compute_facet_data(pts_minus)
            if valid_minus and len(radii_minus) > 0:
                max_minus = np.max(radii_minus)
                shifted_minus = radii_minus - max_minus
                weights_minus = np.exp(shifted_minus / temperature)
                weights_minus /= np.sum(weights_minus)
                smooth_minus = max_minus + temperature * np.log(np.sum(weights_minus))
            else:
                smooth_minus = np.pi
            
            grad[i] += (smooth_plus - smooth_minus) / (2 * eps) * tangent
    
    return smooth_max, grad

def _project_to_sphere(points):
    """Project points to unit sphere."""
    norms = np.linalg.norm(points, axis=1)
    norms[norms < 1e-14] = 1.0
    return points / norms[:, np.newaxis]

def _repulsion_relax(points, max_iter=50, lr=0.01):
    """Repulsion-based relaxation: move points away from each other."""
    points = _normalize(points.copy())
    
    for it in range(max_iter):
        # Compute repulsion forces
        forces = np.zeros_like(points)
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                diff = points[i] - points[j]
                dist_sq = np.dot(diff, diff)
                dist = np.sqrt(dist_sq)
                if dist < 1e-10:
                    continue
                # Repulsion force proportional to 1/dist^2
                force = diff / (dist * dist_sq)
                forces[i] += force
        
        # Project forces to tangent space and apply
        new_points = points.copy()
        for i in range(N):
            tangent_force = forces[i] - np.dot(forces[i], points[i]) * points[i]
            new_points[i] = points[i] + lr * tangent_force
            new_points[i] = new_points[i] / np.linalg.norm(new_points[i])
        
        points = new_points
    
    return points

def _lloyd_iteration(points, max_iter=100, tol=1e-12):
    """Lloyd's algorithm: move each point to the centroid of its Voronoi cell."""
    from scipy.spatial import SphericalVoronoi
    
    points = _normalize(points.copy())
    
    for it in range(max_iter):
        try:
            sv = SphericalVoronoi(points, radius=1.0)
            sv.sort_vertices_of_regions()
            
            new_points = np.zeros_like(points)
            for i in range(N):
                region = sv.regions[i]
                cell_verts = sv.vertices[region]
                if len(cell_verts) == 0:
                    new_points[i] = points[i]
                    continue
                vnorms = np.linalg.norm(cell_verts, axis=1)
                cell_verts = cell_verts / vnorms[:, np.newaxis]
                centroid = np.mean(cell_verts, axis=0)
                cnorm = np.linalg.norm(centroid)
                if cnorm < 1e-10:
                    new_points[i] = points[i]
                else:
                    new_points[i] = centroid / cnorm
            
            new_points = _normalize(new_points)
            
            diff = np.linalg.norm(new_points - points)
            points = new_points
            if diff < tol:
                break
        except Exception:
            break
    
    return points

def _lbfgs_stage(points, max_iter=200, temperature=0.01):
    """L-BFGS optimization using the smooth max radius objective."""
    x0 = points.flatten()
    
    def objective(x):
        pts = _normalize(x.reshape(N, 3))
        val, _ = _soft_max_radius_and_grad(pts, temperature)
        return val
    
    def gradient(x):
        pts = _normalize(x.reshape(N, 3))
        _, grad = _soft_max_radius_and_grad(pts, temperature)
        return grad.flatten()
    
    result = minimize(objective, x0, jac=gradient, method='L-BFGS-B',
                      options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10})
    
    best_points = _normalize(result.x.reshape(N, 3))
    return best_points

def _local_polish(points, max_iter=50):
    """Local polish: alternate between Lloyd and L-BFGS with decreasing temperature."""
    points = _normalize(points.copy())
    
    for temp in [0.1, 0.05, 0.02, 0.01, 0.005]:
        points = _lbfgs_stage(points, max_iter=50, temperature=temp)
        points = _lloyd_iteration(points, max_iter=30, tol=1e-12)
    
    return points

def _facet_targeted_move(points, max_iter=30):
    """Targeted move: directly move vertices of the worst facet toward its circumcenter."""
    points = _normalize(points.copy())
    
    for it in range(max_iter):
        facets, facet_centers, facet_radii, valid = _compute_facet_data(points)
        if not valid or len(facet_radii) == 0:
            break
        
        # Find the worst facet
        worst_idx = np.argmax(facet_radii)
        worst_radius = facet_radii[worst_idx]
        
        if worst_radius < 0.32:  # ~18.3 degrees in radians
            break
        
        worst_facet = facets[worst_idx]
        center = facet_centers[worst_idx]
        
        # Move each vertex of the worst facet slightly toward the center
        step = 0.01
        for vi in worst_facet:
            direction = center - points[vi]
            dnorm = np.linalg.norm(direction)
            if dnorm < 1e-10:
                continue
            direction /= dnorm
            # Project to tangent space
            tangent = direction - np.dot(direction, points[vi]) * points[vi]
            tnorm = np.linalg.norm(tangent)
            if tnorm < 1e-10:
                continue
            tangent /= tnorm
            points[vi] = points[vi] + step * tangent
            points[vi] = points[vi] / np.linalg.norm(points[vi])
        
        points = _normalize(points)
    
    return points

def _fibsphere(n, seed=0):
    """Generate n points on the sphere using Fibonacci spiral."""
    rng = np.random.RandomState(seed)
    i = np.arange(n) + 0.5 + rng.uniform(-0.05, 0.05, size=n)
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i + rng.uniform(-0.05, 0.05, size=n)
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere(n, seed=0):
    """Generate n random points on the sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    return _normalize(pts)

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit
    
    best_points = None
    best_radius = np.pi  # 180 degrees in radians
    
    # Generate diverse initial configurations
    initial_configs = []
    
    # Fibonacci sphere with different seeds
    for seed in range(5):
        initial_configs.append(_fibsphere(N, seed=seed))
    
    # Random configurations
    for seed in range(5):
        initial_configs.append(_random_sphere(N, seed=seed))
    
    # Try each initial configuration
    for idx, init_pts in enumerate(initial_configs):
        if time.time() > deadline:
            break
        
        # Stage 1: Repulsion relaxation
        relaxed = _repulsion_relax(init_pts, max_iter=30, lr=0.01)
        
        # Stage 2: Lloyd iteration
        relaxed = _lloyd_iteration(relaxed, max_iter=50, tol=1e-12)
        
        # Stage 3: L-BFGS polish
        if time.time() < deadline - 10.0:
            relaxed = _local_polish(relaxed)
        
        # Stage 4: Facet targeted move
        if time.time() < deadline - 5.0:
            relaxed = _facet_targeted_move(relaxed)
        
        # Compute radius
        radius = _compute_covering_radius(relaxed)
        
        if radius < best_radius:
            best_radius = radius
            best_points = relaxed.copy()
            elapsed = time.time() - start_time
            print(f"Config {idx}: radius = {np.degrees(radius):.6f} degrees (elapsed: {elapsed:.1f}s)")
        
        # If we have time and this is a good candidate, do more polishing
        if time.time() < deadline - 15.0 and radius < best_radius + 0.01:
            relaxed2 = _local_polish(relaxed)
            radius2 = _compute_covering_radius(relaxed2)
            if radius2 < best_radius:
                best_radius = radius2
                best_points = relaxed2.copy()
                print(f"Config {idx} (refined): radius = {np.degrees(radius2):.6f} degrees")
    
    # Final check
    if best_points is None:
        best_points = _fibsphere(N, seed=0)
    
    # Normalize final result
    best_points = _normalize(best_points)
    
    return best_points
