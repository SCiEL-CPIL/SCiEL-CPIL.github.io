"""Heilbronn triangles: maximize min triangle area for 11 points in unit-area equilateral triangle."""
import numpy as np
import random
import time
from itertools import combinations

N = 11
AREA = np.sqrt(3) / 4.0  # area of equilateral triangle with side 1
H = np.sqrt(3) / 2.0     # height of the triangle

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, H])


def _barycentric_to_cartesian(b):
    """Convert barycentric coordinates (u,v,w) to Cartesian."""
    u, v, w = b
    x = u * V0[0] + v * V1[0] + w * V2[0]
    y = u * V0[1] + v * V1[1] + w * V2[1]
    return np.array([x, y])


def _cartesian_to_barycentric(p):
    """Convert Cartesian point to barycentric coordinates."""
    x, y = p
    # Solve for u, v, w such that p = u*V0 + v*V1 + w*V2, u+v+w=1
    # Using the standard formula
    w = y / H
    # x = v*1 + w*0.5 => v = x - 0.5*w
    v = x - 0.5 * w
    u = 1.0 - v - w
    return np.array([u, v, w])


def _is_inside_triangle(p, eps=1e-9):
    """Check if point p is inside or on the boundary of the triangle."""
    b = _cartesian_to_barycentric(p)
    return all(bi >= -eps for bi in b)


def _clamp_to_triangle(p, eps=1e-6):
    """Clamp point to be inside the triangle."""
    b = _cartesian_to_barycentric(p)
    b = np.maximum(b, 0.0)
    s = b.sum()
    if s < 1e-12:
        return (V0 + V1 + V2) / 3.0
    b = b / s
    return _barycentric_to_cartesian(b)


def _min_triangle_area(points):
    """Compute the minimum area of any triangle formed by 3 of the points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                area = 0.5 * abs((points[j][0] - points[i][0]) * (points[k][1] - points[i][1])
                                 - (points[j][1] - points[i][1]) * (points[k][0] - points[i][0]))
                if area < min_area:
                    min_area = area
    return min_area


def _min_triangle_area_normalized(points):
    """Return min triangle area divided by triangle area."""
    return _min_triangle_area(points) / AREA


def _objective(x):
    """Objective for minimization: negative of min triangle area (we maximize min area)."""
    points = x.reshape(-1, 2)
    # Clamp to triangle
    points = np.array([_clamp_to_triangle(p) for p in points])
    min_area = _min_triangle_area(points)
    return -min_area


def _local_optimize(points, max_iter=500, tol=1e-10):
    """Use scipy minimize to locally optimize the configuration."""
    from scipy.optimize import minimize
    
    x0 = points.flatten()
    
    # Use Nelder-Mead as it doesn't require gradients and works well for this problem
    result = minimize(_objective, x0, method='Nelder-Mead', 
                      options={'maxiter': max_iter, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True})
    
    points_opt = result.x.reshape(-1, 2)
    points_opt = np.array([_clamp_to_triangle(p) for p in points_opt])
    return points_opt


def _generate_initial_config(rng, strategy='structured'):
    """Generate an initial configuration of 11 points."""
    if strategy == 'structured':
        # Use a structured grid-like approach in barycentric coordinates
        # Place points in a pattern that spreads them out
        points = []
        
        # Corners
        points.append(V0 + 0.02 * (V1 - V0))
        points.append(V1 - 0.02 * (V1 - V0))
        points.append(V2 + 0.02 * (V0 - V2))
        
        # Midpoints of edges
        points.append((V0 + V1) / 2.0)
        points.append((V1 + V2) / 2.0)
        points.append((V2 + V0) / 2.0)
        
        # Center
        points.append((V0 + V1 + V2) / 3.0)
        
        # Additional points
        points.append(V0 + 0.3 * (V1 - V0) + 0.1 * (V2 - V0))
        points.append(V0 + 0.1 * (V1 - V0) + 0.3 * (V2 - V0))
        points.append(V1 + 0.1 * (V2 - V1) + 0.1 * (V0 - V1))
        
        return np.array(points)
    elif strategy == 'random':
        points = []
        while len(points) < N:
            u, v = rng.random(), rng.random()
            if u + v < 0.98 and u > 0.01 and v > 0.01:
                p = [u + 0.5 * v, H * v]
                points.append(p)
        return np.array(points)
    elif strategy == 'grid':
        # Create a grid in barycentric space
        points = []
        # Use a 3x3 grid in barycentric coordinates, picking 11 points
        grid_u = [0.05, 0.35, 0.65, 0.95]
        grid_v = [0.05, 0.35, 0.65, 0.95]
        candidates = []
        for u in grid_u:
            for v in grid_v:
                w = 1.0 - u - v
                if w >= -0.01:
                    w = max(w, 0.0)
                    candidates.append(_barycentric_to_cartesian(np.array([u, v, w])))
        
        # Select 11 points from candidates, trying to maximize spread
        rng.shuffle(candidates)
        for p in candidates:
            if len(points) >= N:
                break
            points.append(p)
        
        # If we don't have enough, add random points
        while len(points) < N:
            u, v = rng.random(), rng.random()
            if u + v < 0.98 and u > 0.01 and v > 0.01:
                points.append([u + 0.5 * v, H * v])
        
        return np.array(points)
    else:
        return _generate_initial_config(rng, 'random')


def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin
    
    rng = random.Random(42)
    
    best_points = None
    best_score = -float('inf')
    
    # Try multiple strategies and restarts
    strategies = ['structured', 'grid', 'random']
    
    # First, try the structured approach with local optimization
    for strategy in strategies:
        if time.time() > deadline - 30:
            break
        
        for restart in range(5):
            if time.time() > deadline - 30:
                break
            
            points = _generate_initial_config(rng, strategy)
            
            # Local optimization
            points_opt = _local_optimize(points, max_iter=300)
            score = _min_triangle_area_normalized(points_opt)
            
            if score > best_score:
                best_score = score
                best_points = points_opt.copy()
    
    # If we have time, do more fine-tuning on the best solution
    if best_points is not None and time.time() < deadline - 20:
        # Try to further optimize the best solution with more iterations
        points_fine = _local_optimize(best_points, max_iter=1000)
        score_fine = _min_triangle_area_normalized(points_fine)
        if score_fine > best_score:
            best_score = score_fine
            best_points = points_fine.copy()
    
    # Final check: ensure all points are inside the triangle
    if best_points is not None:
        best_points = np.array([_clamp_to_triangle(p) for p in best_points])
    
    return best_points


# EVOLVE-BLOCK-START
# (The construct function is defined above, but we keep this block for compatibility)
# EVOLVE-BLOCK-END
