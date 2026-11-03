import numpy as np
from scipy.spatial import Delaunay
from scipy.optimize import minimize
import time

N = 50

def _fibonacci_points(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _covering_radius(points):
    """
    Calculate the covering radius of the given points on the unit sphere.
    points: (N, 3) array of unit vectors.
    Returns: covering radius in radians.
    """
    # Normalize points to ensure they are on the sphere
    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    
    try:
        # Compute Delaunay triangulation
        # Delaunay in 3D for points on a sphere gives the convex hull facets
        tri = Delaunay(points)
    except Exception:
        # If Delaunay fails, return a large value
        return np.pi

    max_rad = 0.0
    for simplex in tri.simplices:
        A = points[simplex[0]]
        B = points[simplex[1]]
        C = points[simplex[2]]
        
        # Compute circumcenter of the spherical triangle
        # Vector normal to the plane containing A, B, C
        # N = (B-A) x (C-A)
        N = np.cross(B - A, C - A)
        norm_N = np.linalg.norm(N)
        
        if norm_N < 1e-12:
            # Degenerate triangle
            # Estimate radius by max distance between vertices
            d_ab = np.arccos(np.clip(np.dot(A, B), -1.0, 1.0))
            d_bc = np.arccos(np.clip(np.dot(B, C), -1.0, 1.0))
            d_ca = np.arccos(np.clip(np.dot(C, A), -1.0, 1.0))
            rad = max(d_ab, d_bc, d_ca) / 2.0 # Rough estimate
            if rad > max_rad:
                max_rad = rad
            continue
            
        C_sph = N / norm_N
        # Check if C_sph is the "correct" circumcenter (inside the triangle)
        # The circumcenter should be closer to the centroid than the antipode
        centroid = (A + B + C) / 3.0
        centroid = centroid / np.linalg.norm(centroid)
        
        if np.dot(C_sph, centroid) < 0:
            C_sph = -C_sph
            
        # Angular distance from C_sph to A (or B or C, they are equal)
        dot_val = np.dot(C_sph, A)
        dot_val = np.clip(dot_val, -1.0, 1.0)
        rad = np.arccos(dot_val)
        
        if rad > max_rad:
            max_rad = rad
            
    return max_rad

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s safety margin
    
    # Initial guess: Fibonacci spiral
    x0 = _fibonacci_points(N).flatten()
    
    def objective(x):
        # Check time
        if time.time() > deadline:
            return np.inf
        pts = x.reshape(N, 3)
        # Normalize
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        pts = pts / norms
        return _covering_radius(pts)
    
    # Optimize
    best_x = x0
    best_val = objective(x0)
    
    # Use L-BFGS-B
    # maxiter is a soft limit, we check time inside objective
    res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 200, 'maxfun': 10000})
    
    if res.success and res.fun < best_val:
        best_x = res.x
        best_val = res.fun
        
    # If we have time, try a few more restarts or different initializations?
    # For now, just return the best found.
    
    # Final normalization
    best_pts = best_x.reshape(N, 3)
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    best_pts = best_pts / norms
    
    return best_pts
