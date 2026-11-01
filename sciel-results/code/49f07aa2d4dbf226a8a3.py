"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
import time

def _fibonacci_sphere(n):
    """Generate n points on the unit sphere using the Fibonacci lattice."""
    points = np.zeros((n, 3))
    phi = np.pi * (3.0 - np.sqrt(5.0))  # Golden angle
    for i in range(n):
        y = 1 - (i / float(n - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y
        theta = phi * i  # golden angle increment
        points[i, 0] = np.cos(theta) * radius
        points[i, 1] = y
        points[i, 2] = np.sin(theta) * radius
    return points

def _compute_covering_radius(points):
    """
    Compute the covering radius of the point configuration.
    Returns the maximum angular distance from any point on the sphere to the nearest point.
    This is equivalent to the maximum circumradius of the convex hull facets.
    """
    points = np.asarray(points)
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi
    
    max_radius = 0.0
    
    for simplex in hull.simplices:
        verts = points[simplex]
        if len(verts) != 3:
            continue
            
        # Compute pole of the facet
        e1 = verts[1] - verts[0]
        e2 = verts[2] - verts[0]
        pole = np.cross(e1, e2)
        pole_norm = np.linalg.norm(pole)
        if pole_norm < 1e-12:
            continue
        pole = pole / pole_norm
        
        # Ensure pole is outward pointing
        centroid = np.mean(verts, axis=0)
        if np.dot(pole, centroid) < 0:
            pole = -pole
            
        # Angular distance from pole to any vertex
        cos_theta = np.clip(np.dot(verts[0], pole), -1.0, 1.0)
        theta = np.arccos(cos_theta)
        
        if theta > max_radius:
            max_radius = theta
            
    return max_radius

def construct():
    # Initialize with Fibonacci sphere
    points = _fibonacci_sphere(N)
    
    # Relaxation parameters
    n_iterations = 5000
    step_size = 0.5
    decay_rate = 0.995
    rng = np.random.default_rng(42)
    
    best_radius = np.pi
    best_points = points.copy()
    
    for i in range(n_iterations):
        try:
            hull = ConvexHull(points)
        except Exception:
            # If hull fails, break or reset
            break
            
        # Build adjacency list from hull simplices
        # Each simplex is a triangle of 3 points.
        # We want to find the neighbors for each point.
        neighbors = [[] for _ in range(N)]
        
        for simplex in hull.simplices:
            v0, v1, v2 = simplex
            neighbors[v0].extend([v1, v2])
            neighbors[v1].extend([v0, v2])
            neighbors[v2].extend([v0, v1])
            
        # Calculate targets and update points
        new_points = np.zeros_like(points)
        
        for i in range(N):
            if len(neighbors[i]) == 0:
                # Isolated point (should not happen with convex hull of N>3)
                new_points[i] = points[i]
                continue
                
            # Target is the normalized sum of neighbor points
            # This approximates the spherical centroid of the Voronoi cell
            neighbor_sum = np.sum(points[neighbors[i]], axis=0)
            norm = np.linalg.norm(neighbor_sum)
            if norm < 1e-12:
                target = points[i]
            else:
                target = neighbor_sum / norm
                
            # Move towards target
            new_points[i] = points[i] + step_size * (target - points[i])
            
        # Normalize
        norms = np.linalg.norm(new_points, axis=1, keepdims=True)
        new_points = new_points / norms
        
        # Check for improvement
        current_radius = _compute_covering_radius(new_points)
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = new_points.copy()
            
        # Update points
        points = new_points
        
        # Decay step size
        step_size *= decay_rate
        
        # Occasionally add small random perturbations to escape local minima
        if i % 100 == 50 and step_size > 0.01:
            perturb = rng.standard_normal((N, 3)) * 0.05
            points = points + perturb
            points = points / np.linalg.norm(points, axis=1, keepdims=True)
            
    # Final cleanup: ensure unit norm
    best_points = best_points / np.linalg.norm(best_points, axis=1, keepdims=True)
    
    return best_points
# EVOLVE-BLOCK-END
