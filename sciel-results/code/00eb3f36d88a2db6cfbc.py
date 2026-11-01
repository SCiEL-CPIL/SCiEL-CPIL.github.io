"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 50

def _covering_radius(points):
    """Compute the covering radius in degrees for a set of points on the unit sphere."""
    # The covering radius is the maximum over all points y on the sphere of the angular
    # distance to the nearest point in the configuration.
    # This is equivalent to half the maximum angular distance between any two points
    # in the convex hull's Delaunay triangulation (the largest circumradius of the facets).
    # A practical computation: sample the sphere densely and find the max distance to nearest point.
    # But for a more exact approach, we can use the fact that the covering radius equals
    # the maximum circumradius of the spherical Delaunay triangles.
    # 
    # For efficiency, let's use a dense sampling approach with adaptive refinement.
    # Actually, the most reliable way is to compute the spherical Delaunay triangulation
    # and find the largest circumcircle radius.
    #
    # Let's use scipy's ConvexHull on the 3D points to get the triangular facets,
    # then for each facet compute the circumradius of the spherical triangle.
    
    from scipy.spatial import ConvexHull
    from scipy.spatial import SphericalVoronoi
    
    # Normalize points
    norms = np.linalg.norm(points, axis=1)
    points = points / norms[:, np.newaxis]
    
    # Check for degeneracies
    if np.any(norms < 1e-10):
        return 180.0  # Invalid
    
    # Compute spherical Voronoi to get the covering radius
    try:
        sv = SphericalVoronoi(points, radius=1.0)
        sv.sort_vertices_of_regions()
        
        # The covering radius is the maximum over all Voronoi vertices of the angular
        # distance to the nearest point. But actually, the covering radius is the maximum
        # distance from any point on the sphere to the nearest configuration point.
        # This equals the maximum circumradius of the Delaunay triangles.
        
        # Alternative: for each Voronoi vertex, the distance to the nearest point
        # gives a lower bound. The actual covering radius is the max over the sphere.
        
        # Let's compute it by checking all Voronoi vertices
        vertices = sv.vertices
        # Normalize vertices
        vnorms = np.linalg.norm(vertices, axis=1)
        vertices = vertices / vnorms[:, np.newaxis]
        
        # For each vertex, find the nearest point
        # Use dot product for angular distance
        dots = vertices @ points.T  # (n_vertices, N)
        min_dots = np.max(dots, axis=1)  # max dot = min angular distance
        # Angular distance = arccos(dot)
        angles = np.arccos(np.clip(min_dots, -1.0, 1.0))
        
        return np.degrees(np.max(angles))
    except Exception:
        # Fallback: dense sampling
        # Generate a dense set of points on the sphere
        n_samples = 50000
        rng = np.random.RandomState(42)
        # Use Fibonacci sphere for uniform sampling
        i = np.arange(n_samples) + 0.5
        z = 1.0 - 2.0 * i / n_samples
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        sample_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        
        # For each sample point, find the nearest configuration point
        dots = sample_points @ points.T
        min_dots = np.max(dots, axis=1)
        angles = np.arccos(np.clip(min_dots, -1.0, 1.0))
        
        return np.degrees(np.max(angles))


def _objective(x):
    """Objective function: covering radius in degrees."""
    points = x.reshape(N, 3)
    # Normalize
    norms = np.linalg.norm(points, axis=1)
    if np.any(norms < 1e-10):
        return 180.0
    points = points / norms[:, np.newaxis]
    return _covering_radius(points)


def _fibsphere(n, seed=0):
    """Generate n points on the sphere using Fibonacci spiral."""
    rng = np.random.RandomState(seed)
    # Add small perturbation for diversity
    i = np.arange(n) + 0.5 + rng.uniform(-0.1, 0.1, size=n)
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i + rng.uniform(-0.1, 0.1, size=n)
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(n, seed=0):
    """Generate n random points on the sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    norms = np.linalg.norm(pts, axis=1)
    return pts / norms[:, np.newaxis]


def _lloyd_relaxation(points, max_iter=100, tol=1e-10):
    """Lloyd's algorithm on the sphere: iteratively move each point to the centroid
    of its Voronoi cell (spherical mean of points in its cell)."""
    from scipy.spatial import SphericalVoronoi
    
    points = points.copy()
    norms = np.linalg.norm(points, axis=1)
    points = points / norms[:, np.newaxis]
    
    for it in range(max_iter):
        try:
            sv = SphericalVoronoi(points, radius=1.0)
            sv.sort_vertices_of_regions()
            
            new_points = np.zeros_like(points)
            for i in range(N):
                region = sv.regions[i]
                # Get the vertices of this Voronoi cell
                cell_verts = sv.vertices[region]
                if len(cell_verts) == 0:
                    new_points[i] = points[i]
                    continue
                # Spherical mean: average the normalized vectors
                vnorms = np.linalg.norm(cell_verts, axis=1)
                cell_verts = cell_verts / vnorms[:, np.newaxis]
                centroid = np.mean(cell_verts, axis=0)
                cnorm = np.linalg.norm(centroid)
                if cnorm < 1e-10:
                    new_points[i] = points[i]
                else:
                    new_points[i] = centroid / cnorm
            new_points = new_points / np.linalg.norm(new_points, axis=1)[:, np.newaxis]
            
            # Check convergence
            diff = np.linalg.norm(new_points - points)
            points = new_points
            if diff < tol:
                break
        except Exception:
            break
    
    return points


def _gradient_descent(points, max_iter=200, lr=0.01):
    """Simple gradient descent using finite differences on the sphere."""
    from scipy.spatial import SphericalVoronoi
    
    points = points.copy()
    norms = np.linalg.norm(points, axis=1)
    points = points / norms[:, np.newaxis]
    
    def _local_radius(pts):
        try:
            sv = SphericalVoronoi(pts, radius=1.0)
            sv.sort_vertices_of_regions()
            vertices = sv.vertices
            vnorms = np.linalg.norm(vertices, axis=1)
            vertices = vertices / vnorms[:, np.newaxis]
            dots = vertices @ pts.T
            min_dots = np.max(dots, axis=1)
            angles = np.arccos(np.clip(min_dots, -1.0, 1.0))
            return np.degrees(np.max(angles))
        except Exception:
            return 180.0
    
    for it in range(max_iter):
        obj = _local_radius(points)
        if obj < 18.31:
            break
        
        # Compute gradient via finite differences
        grad = np.zeros_like(points)
        eps = 1e-5
        for i in range(N):
            for j in range(3):
                # Tangent direction
                e = np.zeros(3)
                e[j] = 1.0
                # Project to tangent space
                tangent = e - np.dot(e, points[i]) * points[i]
                tnorm = np.linalg.norm(tangent)
                if tnorm < 1e-10:
                    continue
                tangent /= tnorm
                
                # Perturb
                pts_plus = points.copy()
                pts_plus[i] = points[i] + eps * tangent
                pts_plus[i] /= np.linalg.norm(pts_plus[i])
                
                pts_minus = points.copy()
                pts_minus[i] = points[i] - eps * tangent
                pts_minus[i] /= np.linalg.norm(pts_minus[i])
                
                grad[i] += (np.dot(tangent, np.zeros(3))) * (_local_radius(pts_plus) - _local_radius(pts_minus)) / (2 * eps)
                # Actually, we need the gradient in the full 3D space projected
                # Let's just do a simpler approach: use the tangent direction
        
        # This is getting complicated. Let's use a simpler approach.
        break
    
    return points


def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    best_points = None
    best_radius = 180.0
    
    # Generate diverse initial configurations
    initial_configs = []
    
    # Fibonacci sphere with different seeds
    for seed in range(10):
        initial_configs.append(_fibsphere(N, seed=seed))
    
    # Random configurations
    for seed in range(10):
        initial_configs.append(_random_sphere(N, seed=seed))
    
    # Try each initial configuration with Lloyd relaxation
    for idx, init_pts in enumerate(initial_configs):
        if time.time() > deadline:
            break
        
        # Lloyd relaxation
        relaxed = _lloyd_relaxation(init_pts, max_iter=50, tol=1e-10)
        
        # Compute radius
        radius = _covering_radius(relaxed)
        
        if radius < best_radius:
            best_radius = radius
            best_points = relaxed.copy()
            print(f"Config {idx}: radius = {radius:.6f} degrees")
        
        # If we have time, try a second round of Lloyd with tighter tolerance
        if time.time() < deadline - 5.0 and radius < best_radius + 0.1:
            relaxed2 = _lloyd_relaxation(relaxed, max_iter=100, tol=1e-12)
            radius2 = _covering_radius(relaxed2)
            if radius2 < best_radius:
                best_radius = radius2
                best_points = relaxed2.copy()
                print(f"Config {idx} (refined): radius = {radius2:.6f} degrees")
    
    # Final check
    if best_points is None:
        best_points = _fibsphere(N, seed=0)
    
    # Normalize final result
    norms = np.linalg.norm(best_points, axis=1)
    best_points = best_points / norms[:, np.newaxis]
    
    return best_points
