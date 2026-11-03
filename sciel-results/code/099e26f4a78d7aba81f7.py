import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130

def _compute_covering_radius(points):
    """
    Compute the covering radius of a set of points on the unit sphere.
    """
    try:
        hull = ConvexHull(points)
        facets = hull.simplices
        max_radius = 0.0
        
        # Vectorized circumcenter calculation
        p1 = points[facets[:, 0]]
        p2 = points[facets[:, 1]]
        p3 = points[facets[:, 2]]
        
        v1 = p2 - p1
        v2 = p3 - p1
        cross = np.cross(v1, v2)
        norms = np.linalg.norm(cross, axis=1, keepdims=True)
        
        # Avoid division by zero
        norms[norms < 1e-12] = 1e-12
        c = cross / norms
        
        # Ensure correct orientation (dot with p1 > 0)
        dots = np.sum(c * p1, axis=1)
        flip = dots < 0
        c[flip] *= -1
        
        # Calculate distances
        dots_c = np.sum(c * p1, axis=1)
        dots_c = np.clip(dots_c, -1.0, 1.0)
        dists = np.arccos(dots_c)
        
        return np.degrees(np.max(dists))
    except Exception:
        return 180.0

def construct():
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Normalize
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    
    start_time = time.time()
    deadline = start_time + 140.0 # 10s safety margin
    
    best_points = points.copy()
    best_radius = _compute_covering_radius(points)
    
    # Lloyd's Algorithm
    while time.time() < deadline:
        try:
            hull = ConvexHull(points)
            facets = hull.simplices
        except Exception:
            break
            
        # Calculate circumcenters for all facets
        p1 = points[facets[:, 0]]
        p2 = points[facets[:, 1]]
        p3 = points[facets[:, 2]]
        
        v1 = p2 - p1
        v2 = p3 - p1
        cross = np.cross(v1, v2)
        norms = np.linalg.norm(cross, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1e-12
        c = cross / norms
        
        # Fix orientation
        dots = np.sum(c * p1, axis=1)
        flip = dots < 0
        c[flip] *= -1
        
        # Accumulate circumcenters for each vertex
        sums = np.zeros((N, 3))
        for idx, tri in enumerate(facets):
            c_vec = c[idx]
            sums[tri[0]] += c_vec
            sums[tri[1]] += c_vec
            sums[tri[2]] += c_vec
            
        # Normalize sums to get new positions
        norms_sums = np.linalg.norm(sums, axis=1, keepdims=True)
        norms_sums[norms_sums < 1e-12] = 1e-12
        new_points = sums / norms_sums
        
        # Check for non-finite values
        if not np.all(np.isfinite(new_points)):
            break
            
        points = new_points
        
        # Periodic evaluation
        current_radius = _compute_covering_radius(points)
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = points.copy()
            
    return best_points
