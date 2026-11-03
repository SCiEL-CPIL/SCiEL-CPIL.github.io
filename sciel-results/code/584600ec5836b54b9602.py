import numpy as np
from scipy.spatial import SphericalVoronoi, ConvexHull
from scipy.optimize import minimize
import time

N = 130

def _compute_covering_radius(points):
    """
    Computes the covering radius for points on the unit sphere.
    The covering radius is the maximum distance from any point on the sphere
    to the nearest point in 'points'.
    This is equal to the maximum circumradius of the spherical Voronoi cells.
    """
    try:
        vor = SphericalVoronoi(points)
        vertices = vor.vertices
        dots = vertices @ points.T
        dots = np.clip(dots, -1.0, 1.0)
        min_dots = np.max(dots, axis=1)
        min_dists = np.arccos(min_dots)
        return np.max(min_dists)
    except Exception:
        return np.pi / 2.0

def _compute_facet_circumradii_and_centers(points):
    """
    Compute circumradii and circumcenters of all Delaunay triangles (facets).
    Returns:
        circumradii: array of shape (M,) with angular circumradius of each facet
        circumcenters: array of shape (M, 3) with circumcenter of each facet
        facet_indices: array of shape (M, 3) with indices of points forming each facet
    """
    try:
        vor = SphericalVoronoi(points)
        # Get Delaunay triangles from Voronoi
        # Each Voronoi vertex corresponds to a Delaunay triangle
        # We need to find which 3 sites are equidistant to each Voronoi vertex
        
        vertices = vor.vertices
        M = len(vertices)
        
        # For each vertex, find the 3 nearest sites (should be equidistant)
        dots = vertices @ points.T  # (M, N)
        dots = np.clip(dots, -1.0, 1.0)
        
        # Get the 3 largest dot products (closest points)
        # Use argpartition for efficiency
        top3_idx = np.argpartition(dots, -3, axis=1)[:, -3:]  # (M, 3)
        
        # Sort the top 3 by dot product to ensure consistent ordering
        top3_dots = np.take_along_axis(dots, top3_idx, axis=1)
        sort_order = np.argsort(-top3_dots, axis=1)  # descending
        top3_idx = np.take_along_axis(top3_idx, sort_order, axis=1)
        
        # Circumradius is arccos of the dot product (they should be approximately equal)
        # Use the average of the 3 dot products for robustness
        avg_dot = np.mean(top3_dots, axis=1)
        avg_dot = np.clip(avg_dot, -1.0, 1.0)
        circumradii = np.arccos(avg_dot)
        
        return circumradii, vertices, top3_idx
    except Exception:
        return None, None, None

def _facet_based_optimize(points, max_iters=200, step_size=0.05, deadline=None):
    """
    Optimize points by moving them to equalize circumradii of adjacent facets.
    Larger circumradius facets push their vertices away, smaller pull them closer.
    """
    points = points.copy()
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
            
        circumradii, circumcenters, facet_indices = _compute_facet_circumradii_and_centers(points)
        if circumradii is None:
            break
            
        # Compute force on each point
        # For each point, sum up contributions from all facets it belongs to
        # If a facet has large circumradius, push its vertices outward (away from circumcenter)
        # If small, pull inward
        
        forces = np.zeros_like(points)
        M = len(circumradii)
        
        # Mean circumradius for normalization
        mean_r = np.mean(circumradii)
        
        for m in range(M):
            r = circumradii[m]
            # Weight: how much this facet deviates from mean
            weight = (r - mean_r) * step_size
            
            # Get the 3 points of this facet
            idxs = facet_indices[m]
            cc = circumcenters[m]
            
            # For each vertex of this facet, compute direction away/toward circumcenter
            for i in idxs:
                # Direction from circumcenter to point i
                d = points[i] - cc
                d_norm = np.linalg.norm(d)
                if d_norm < 1e-10:
                    continue
                d = d / d_norm
                
                # Add force: if r > mean_r, push away from cc (positive weight)
                # if r < mean_r, pull toward cc (negative weight)
                forces[i] += weight * d
        
        # Normalize forces to have reasonable magnitude
        force_mags = np.linalg.norm(forces, axis=1)
        max_force = np.max(force_mags)
        if max_force > 1e-10:
            forces = forces / max_force * step_size
        
        # Update points: move in direction of force, then re-normalize
        new_points = points + forces
        norms = np.linalg.norm(new_points, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        new_points = new_points / norms
        
        # Check convergence
        diff = np.linalg.norm(new_points - points, axis=1)
        if np.max(diff) < 1e-7:
            break
            
        points = new_points
    
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    best_points = None
    best_radius = np.pi / 2.0
    
    # Multiple restarts with different initializations
    n_restarts = 3
    
    for restart in range(n_restarts):
        if time.time() > deadline - 40.0:
            break
            
        # Different initialization strategies
        if restart == 0:
            # Fibonacci Spiral
            i = np.arange(N) + 0.5
            z = 1.0 - 2.0 * i / N
            phi = np.pi * (1.0 + np.sqrt(5.0)) * i
            r = np.sqrt(1.0 - z * z)
            points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        elif restart == 1:
            # Random initialization with slight perturbation of Fibonacci
            i = np.arange(N) + 0.5
            z = 1.0 - 2.0 * i / N
            phi = np.pi * (1.0 + np.sqrt(5.0)) * i + np.random.randn(N) * 0.1
            r = np.sqrt(1.0 - z * z)
            points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            points = points / norms
        else:
            # Another random initialization
            points = np.random.randn(N, 3)
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            points = points / norms
        
        current_radius = _compute_covering_radius(points)
        
        # Phase 1: Lloyd's Algorithm
        for k in range(30):
            if time.time() > deadline - 30.0:
                break
                
            try:
                vor = SphericalVoronoi(points)
                new_points = np.zeros_like(points)
                for i in range(N):
                    region = vor.regions[i]
                    if not region:
                        new_points[i] = points[i]
                        continue
                    verts = vor.vertices[region]
                    c = np.sum(verts, axis=0)
                    norm = np.linalg.norm(c)
                    if norm < 1e-10:
                        new_points[i] = points[i]
                    else:
                        new_points[i] = c / norm
                
                diff = np.linalg.norm(new_points - points, axis=1)
                if np.max(diff) < 1e-6:
                    break
                    
                points = new_points
                current_radius = _compute_covering_radius(points)
                
            except Exception:
                break
        
        # Phase 2: Facet-based optimization
        if time.time() < deadline - 20.0:
            points = _facet_based_optimize(points, max_iters=100, step_size=0.03, deadline=deadline - 15.0)
            current_radius = _compute_covering_radius(points)
        
        # Phase 3: Nelder-Mead polish
        if time.time() < deadline - 10.0:
            def objective(flat_points):
                if time.time() > deadline - 5.0:
                    return 10.0
                pts = flat_points.reshape(N, 3)
                norms = np.linalg.norm(pts, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                pts = pts / norms
                return _compute_covering_radius(pts)
                
            initial_flat = points.flatten()
            
            try:
                res = minimize(objective, initial_flat, method='Nelder-Mead', 
                               options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
                
                if res.fun < current_radius:
                    current_radius = res.fun
                    best_flat = res.x
                    norms = np.linalg.norm(best_flat.reshape(N, 3), axis=1, keepdims=True)
                    norms[norms == 0] = 1.0
                    points = best_flat.reshape(N, 3) / norms
            except Exception:
                pass
        
        # Update best
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = points.copy()
    
    # If no valid result, use Fibonacci as fallback
    if best_points is None:
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        best_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Final normalization
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    best_points = best_points / norms
    
    return best_points
