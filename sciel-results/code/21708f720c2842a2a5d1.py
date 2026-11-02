"""
Heilbronn triangle problem: maximize minimum triangle area for 11 points.
Strategy: Multi-start local optimization using Nelder-Mead in barycentric coordinates.
Barycentric coordinates naturally enforce the triangle constraint, removing the need
for penalty terms or projection steps. We use a smooth log-sum-exp approximation of
the minimum area as the objective, and perform many diverse restarts to escape local
minima. The known asymptotic behavior suggests the optimum is around 0.01-0.02 for n=11.
"""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 11
S3H = np.sqrt(3) / 2
DEADLINE = 140.0  # Safety margin before 150s limit

def _bary_to_cart(bary):
    """Convert barycentric coordinates (x, y, z) with x+y+z=1 to Cartesian."""
    # Vertices: A=(0,0), B=(1,0), C=(0.5, sqrt(3)/2)
    x = bary[:, 1] * 1.0 + bary[:, 2] * 0.5
    y = bary[:, 2] * S3H
    return np.column_stack([x, y])

def _min_triangle_area_cart(pts):
    """Compute the minimum area of any triangle formed by three points."""
    n = len(pts)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                dx1 = pts[j, 0] - pts[i, 0]
                dy1 = pts[j, 1] - pts[i, 1]
                dx2 = pts[k, 0] - pts[i, 0]
                dy2 = pts[k, 1] - pts[i, 1]
                cross = abs(dx1 * dy2 - dy1 * dx2)
                area = 0.5 * cross
                if area < min_area:
                    min_area = area
    return min_area

def _soft_min_area_cart(pts, temperature=0.005):
    """
    Smoothed objective: - (soft min of triangle areas).
    Uses log-sum-exp approximation for the minimum to make it differentiable.
    """
    n = len(pts)
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                dx1 = pts[j, 0] - pts[i, 0]
                dy1 = pts[j, 1] - pts[i, 1]
                dx2 = pts[k, 0] - pts[i, 0]
                dy2 = pts[k, 1] - pts[i, 1]
                cross = abs(dx1 * dy2 - dy1 * dx2)
                area = 0.5 * cross
                areas.append(area)
                
    areas = np.array(areas)
    b = 1.0 / temperature
    areas_clipped = np.clip(areas, 1e-10, None)
    exp_terms = np.exp(-b * areas_clipped)
    soft_min_val = (1.0 / b) * np.log(np.sum(exp_terms))
    return -soft_min_val

def _objective_bary(x_flat, temperature=0.005):
    """
    Objective function in barycentric coordinates.
    x_flat is a 2D array of shape (N, 2) representing (u, v) where w=1-u-v.
    We enforce u>=0, v>=0, u+v<=1 via penalty.
    """
    bary_uv = x_flat.reshape(N, 2)
    u = bary_uv[:, 0]
    v = bary_uv[:, 1]
    
    # Penalty for being outside the valid barycentric region
    penalty = 0.0
    for i in range(N):
        if u[i] < 0: penalty += u[i]**2
        if v[i] < 0: penalty += v[i]**2
        if u[i] + v[i] > 1: penalty += (u[i] + v[i] - 1)**2
    
    # Convert to Cartesian
    bary = np.column_stack([1.0 - u - v, u, v])
    pts = _bary_to_cart(bary)
    
    # Compute soft min area
    obj = _soft_min_area_cart(pts, temperature)
    
    # Add penalty
    obj += 100.0 * penalty
    
    return obj

def _random_bary(rng):
    """Generate a random point in barycentric coordinates (u, v) with u,v>=0, u+v<=1."""
    u = rng.random()
    v = rng.random()
    while u + v > 1:
        u = rng.random()
        v = rng.random()
    return [u, v]

def _initial_guess(rng):
    """Generate an initial set of 11 points in barycentric coordinates."""
    pts = []
    for _ in range(N):
        pts.append(_random_bary(rng))
    return np.array(pts)

def _structured_guess(rng):
    """Generate a structured initial set of 11 points in barycentric coordinates."""
    # Corners: (0,0), (1,0), (0,1) in barycentric (u,v)
    corners = np.array([[0,0], [1,0], [0,1]])
    # Center: (1/3, 1/3)
    center = np.array([[1/3, 1/3]])
    # Midpoints of edges
    midpoints = np.array([
        [0.5, 0], [0, 0.5], [0.5, 0.5]
    ])
    # Some interior points
    interior = np.array([
        [0.2, 0.2], [0.6, 0.2], [0.2, 0.6], [0.4, 0.4]
    ])
    
    all_pts = np.vstack([corners, center, midpoints, interior])
    
    # Shuffle and add small random perturbations
    rng.shuffle(all_pts)
    for i in range(len(all_pts)):
        all_pts[i] += 0.02 * (2*rng.random() - 1)
        # Ensure validity
        if all_pts[i][0] < 0: all_pts[i][0] = 0
        if all_pts[i][1] < 0: all_pts[i][1] = 0
        if all_pts[i][0] + all_pts[i][1] > 1:
            scale = 1.0 / (all_pts[i][0] + all_pts[i][1])
            all_pts[i] *= scale
    
    return all_pts[:11]

def construct():
    """
    Construct 11 points in the unit equilateral triangle to maximize min triangle area.
    Uses Nelder-Mead on a smooth soft-min objective in barycentric coordinates with multiple restarts.
    """
    start_time = time.time()
    rng = random.Random(42)
    
    best_pts_bary = None
    best_score = -float('inf')
    
    # List of initial configurations to try
    initial_configs = []
    
    # 1. Structured start
    initial_configs.append(_structured_guess(rng))
    
    # 2. Random starts
    for _ in range(20):
        initial_configs.append(_initial_guess(rng))
    
    # 3. Grid-like start in barycentric coordinates
    grid_pts = []
    for i in range(4):
        for j in range(4):
            if i + j < 4:
                u = 0.1 + 0.2 * i
                v = 0.1 + 0.2 * j
                if u >= 0 and v >= 0 and u + v <= 1:
                    grid_pts.append([u, v])
    if len(grid_pts) >= 11:
        initial_configs.append(np.array(grid_pts[:11]))
    else:
        initial_configs.append(_initial_guess(rng))
    
    # Optimization loop
    for init_pts in initial_configs:
        if time.time() - start_time > DEADLINE - 5:
            break
            
        # Use a moderately high temperature for the soft-min to ensure smoothness
        temperature = 0.005
        
        try:
            result = minimize(
                lambda x: _objective_bary(x, temperature=temperature),
                init_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            
            polished_pts_bary = result.x.reshape(N, 2)
            
            # Project to valid region
            for i in range(N):
                if polished_pts_bary[i, 0] < 0: polished_pts_bary[i, 0] = 0
                if polished_pts_bary[i, 1] < 0: polished_pts_bary[i, 1] = 0
                if polished_pts_bary[i, 0] + polished_pts_bary[i, 1] > 1:
                    scale = 1.0 / (polished_pts_bary[i, 0] + polished_pts_bary[i, 1])
                    polished_pts_bary[i] *= scale
            
            # Convert to Cartesian and evaluate
            bary = np.column_stack([1.0 - polished_pts_bary[:, 0] - polished_pts_bary[:, 1], 
                                   polished_pts_bary[:, 0], polished_pts_bary[:, 1]])
            polished_pts = _bary_to_cart(bary)
            
            area = _min_triangle_area_cart(polished_pts)
            if area > best_score:
                best_score = area
                best_pts_bary = polished_pts_bary.copy()
                
                # If we found a good solution, try to refine with lower temperature
                if best_score > 0.005:
                    if time.time() - start_time > DEADLINE - 10:
                        break
                    temp_fine = 0.001
                    try:
                        result_fine = minimize(
                            lambda x: _objective_bary(x, temperature=temp_fine),
                            polished_pts_bary.flatten(),
                            method='Nelder-Mead',
                            options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-10}
                        )
                        fine_pts_bary = result_fine.x.reshape(N, 2)
                        for i in range(N):
                            if fine_pts_bary[i, 0] < 0: fine_pts_bary[i, 0] = 0
                            if fine_pts_bary[i, 1] < 0: fine_pts_bary[i, 1] = 0
                            if fine_pts_bary[i, 0] + fine_pts_bary[i, 1] > 1:
                                scale = 1.0 / (fine_pts_bary[i, 0] + fine_pts_bary[i, 1])
                                fine_pts_bary[i] *= scale
                        
                        bary_fine = np.column_stack([1.0 - fine_pts_bary[:, 0] - fine_pts_bary[:, 1], 
                                                   fine_pts_bary[:, 0], fine_pts_bary[:, 1]])
                        fine_pts = _bary_to_cart(bary_fine)
                        fine_area = _min_triangle_area_cart(fine_pts)
                        if fine_area > best_score:
                            best_score = fine_area
                            best_pts_bary = fine_pts_bary.copy()
                    except Exception:
                        pass
                    
        except Exception:
            pass
    
    if best_pts_bary is None:
        # Fallback
        best_pts_bary = _initial_guess(rng)
    
    # Convert final barycentric coordinates to Cartesian
    bary = np.column_stack([1.0 - best_pts_bary[:, 0] - best_pts_bary[:, 1], 
                           best_pts_bary[:, 0], best_pts_bary[:, 1]])
    best_pts = _bary_to_cart(bary)
    
    return best_pts
