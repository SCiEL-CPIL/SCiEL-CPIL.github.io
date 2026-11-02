"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time
import random

def _min_triangle_area(points):
    """
    Compute the minimum area over all triples of points.
    points: numpy array of shape (N, 2)
    """
    n = len(points)
    min_area = float('inf')
    # Vectorized calculation for speed
    # We can compute all pairwise differences, but triple loops are fine for N=14
    # 364 triangles.
    for i in range(n):
        pi = points[i]
        for j in range(i + 1, n):
            v1 = points[j] - pi
            for k in range(j + 1, n):
                v2 = points[k] - pi
                # Cross product in 2D
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                if area < min_area:
                    min_area = area
    return min_area

def _hull_area(points):
    """
    Compute the area of the convex hull of the points.
    """
    if len(points) < 3:
        return 0.0
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        # If points are collinear or degenerate
        return 1e-12

def _raw_score(points):
    """
    Score = min_triangle_area / convex_hull_area
    """
    h_area = _hull_area(points)
    if h_area < 1e-9:
        return 0.0
    m_area = _min_triangle_area(points)
    return m_area / h_area

def construct():
    start_time = time.time()
    # Leave 10s margin for overhead/return
    deadline = start_time + 140.0
    
    # Initial configurations
    configs = []
    
    # 1. Regular 14-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    x = np.cos(angles)
    y = np.sin(angles)
    configs.append(np.column_stack((x, y)))
    
    # 2. Regular 14-gon with alternating radii (Star-like)
    radii = np.ones(N)
    radii[::2] = 0.8 # Every other point is closer to center
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    configs.append(np.column_stack((x, y)))
    
    # 3. Two concentric 7-gons
    # Outer ring
    angles_out = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    x_out = np.cos(angles_out)
    y_out = np.sin(angles_out)
    # Inner ring (rotated by half step)
    angles_in = np.linspace(np.pi/7, 2 * np.pi + np.pi/7, 7, endpoint=False)
    x_in = 0.5 * np.cos(angles_in)
    y_in = 0.5 * np.sin(angles_in)
    
    # Interleave them? Or just stack?
    # Let's stack and sort by angle to make it a convex-ish shape?
    # No, just place them. The optimizer will fix the positions.
    pts3 = np.vstack([np.column_stack((x_out, y_out)), np.column_stack((x_in, y_in))])
    configs.append(pts3)
    
    # 4. Random points in a circle (Convex distribution)
    rng = np.random.RandomState(42)
    r = np.sqrt(rng.uniform(0, 1, N)) # Uniform in disk
    theta = rng.uniform(0, 2 * np.pi, N)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    configs.append(np.column_stack((x, y)))

    best_points = configs[0].copy()
    best_score = _raw_score(best_points)
    
    # Objective function for minimization
    def objective(flat_points):
        if time.time() > deadline:
            return 1e9
        pts = flat_points.reshape(-1, 2)
        return -_raw_score(pts)
    
    # Optimize each configuration
    for i, init_pts in enumerate(configs):
        if time.time() > deadline - 20:
            break
            
        x0 = init_pts.flatten()
        
        try:
            # Nelder-Mead is robust for non-smooth objectives
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
            
            if res.fun < -best_score:
                new_score = -res.fun
                if new_score > best_score:
                    best_score = new_score
                    best_points = res.x.reshape(-1, 2)
                    
        except Exception:
            pass

    # Final check
    return best_points

# EVOLVE-BLOCK-END
