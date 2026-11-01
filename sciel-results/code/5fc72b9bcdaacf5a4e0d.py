"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

def _covering_radius(points):
    """Compute the covering radius in degrees for points on unit sphere."""
    # Ensure points are on unit sphere
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / norms
    
    # Build convex hull to find facets
    try:
        hull = ConvexHull(points)
    except Exception:
        # Fallback: compute max distance to nearest point on a grid
        theta = np.linspace(0, np.pi, 200)
        phi = np.linspace(0, 2*np.pi, 400)
        theta_grid, phi_grid = np.meshgrid(theta, phi, indexing='ij')
        grid_points = np.column_stack([
            np.sin(theta_grid) * np.cos(phi_grid).ravel(),
            np.sin(theta_grid) * np.sin(phi_grid).ravel(),
            np.cos(theta_grid).ravel()
        ])
        # For each grid point, find nearest configuration point
        min_dists = []
        for gp in grid_points:
            d = np.arccos(np.clip(np.dot(points, gp), -1, 1))
            min_dists.append(d.min())
        return np.degrees(np.max(min_dists))
    
    # For each facet, compute the circumradius (distance from centroid to farthest vertex)
    # The covering radius is the max over facets of the distance from the facet's 
    # "center" (the point farthest from all vertices in the cap) to the vertices
    # More precisely: for each facet, the covering radius contribution is the angular 
    # distance from the facet's center to its vertices
    
    max_radius = 0.0
    for simplex in hull.simplices:
        facet_points = points[simplex]
        # The center of the empty cap is the point on the sphere farthest from all 
        # facet vertices. For a spherical simplex, this is the circumcenter.
        # We can find it by optimizing: maximize min distance to facet vertices
        # subject to being on the unit sphere.
        
        # For a triangle on the sphere, the circumcenter can be found analytically.
        # For higher-dimensional facets, we use optimization.
        
        def neg_min_dist(x):
            x = x / np.linalg.norm(x)
            dists = np.arccos(np.clip(np.dot(facet_points, x), -1, 1))
            return -dists.min()
        
        # Start from the normalized centroid
        centroid = np.mean(facet_points, axis=0)
        centroid = centroid / np.linalg.norm(centroid)
        
        result = minimize(neg_min_dist, centroid, method='BFGS', 
                         options={'maxiter': 100, 'gtol': 1e-8})
        
        if result.success:
            center = result.x / np.linalg.norm(result.x)
            dists = np.arccos(np.clip(np.dot(facet_points, center), -1, 1))
            radius = dists.max()
        else:
            # Fallback: use the centroid
            center = centroid
            dists = np.arccos(np.clip(np.dot(facet_points, center), -1, 1))
            radius = dists.max()
        
        max_radius = max(max_radius, radius)
    
    return np.degrees(max_radius)


def _local_refine(points, worst_facet_indices, max_iter=50):
    """Refine points by moving them to reduce the covering radius."""
    points = points.copy()
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / norms
    
    for _ in range(max_iter):
        # Find the current covering radius and worst facet
        try:
            hull = ConvexHull(points)
        except Exception:
            break
        
        # Find the facet with the largest circumradius
        max_radius = 0.0
        worst_facet = None
        for simplex in hull.simplices:
            facet_points = points[simplex]
            centroid = np.mean(facet_points, axis=0)
            centroid = centroid / np.linalg.norm(centroid)
            dists = np.arccos(np.clip(np.dot(facet_points, centroid), -1, 1))
            radius = dists.max()
            if radius > max_radius:
                max_radius = radius
                worst_facet = simplex
        
        if worst_facet is None:
            break
        
        # Move points in the worst facet slightly outward from the centroid
        centroid = np.mean(points[worst_facet], axis=0)
        centroid = centroid / np.linalg.norm(centroid)
        
        for idx in worst_facet:
            # Move point away from centroid (outward)
            direction = points[idx] - centroid
            direction = direction / np.linalg.norm(direction)
            step = 0.01
            new_point = points[idx] + step * direction
            new_point = new_point / np.linalg.norm(new_point)
            
            # Check if this improves the covering radius
            old_radius = _covering_radius(points)
            points[idx] = new_point
            new_radius = _covering_radius(points)
            
            if new_radius >= old_radius:
                # Revert
                points[idx] = points[idx] - step * direction
                points[idx] = points[idx] / np.linalg.norm(points[idx])
    
    return points


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    best_points = None
    best_radius = np.inf
    
    # Multi-start simulated annealing
    np.random.seed(42)
    
    for restart in range(10):
        if time.time() > deadline:
            break
        
        # Initialize with random points
        points = np.random.randn(N, 3)
        points = points / np.linalg.norm(points, axis=1, keepdims=True)
        
        # Simulated annealing
        temperature = 0.1
        cooling = 0.999
        step_size = 0.1
        accept_rate = 1.0
        
        for iteration in range(5000):
            if time.time() > deadline:
                break
            
            # Pick a random point to move
            idx = np.random.randint(N)
            
            # Generate a random perturbation
            perturbation = np.random.randn(3) * step_size
            new_point = points[idx] + perturbation
            new_point = new_point / np.linalg.norm(new_point)
            
            # Compute old and new covering radii
            old_radius = _covering_radius(points)
            points[idx] = new_point
            new_radius = _covering_radius(points)
            
            # Acceptance criterion
            delta = new_radius - old_radius
            if delta < 0 or np.random.rand() < np.exp(-delta / temperature):
                # Accept
                accept_count = 1
            else:
                # Revert
                points[idx] = points[idx] - perturbation
                points[idx] = points[idx] / np.linalg.norm(points[idx])
                accept_count = 0
            
            # Adaptive step size
            if iteration % 100 == 0 and iteration > 0:
                if accept_rate > 0.5:
                    step_size *= 1.1
                elif accept_rate < 0.2:
                    step_size *= 0.9
                accept_rate = accept_count / 100.0
            
            # Cool down
            temperature *= cooling
            
            # Update best
            current_radius = _covering_radius(points)
            if current_radius < best_radius:
                best_radius = current_radius
                best_points = points.copy()
        
        # Local refinement
        if time.time() < deadline - 10:
            refined = _local_refine(points, max_iter=20)
            refined_radius = _covering_radius(refined)
            if refined_radius < best_radius:
                best_radius = refined_radius
                best_points = refined.copy()
    
    # Ensure we have a valid result
    if best_points is None:
        # Fallback to Fibonacci spiral
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        best_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Final normalization
    best_points = best_points / np.linalg.norm(best_points, axis=1, keepdims=True)
    
    return best_points
# EVOLVE-BLOCK-END
