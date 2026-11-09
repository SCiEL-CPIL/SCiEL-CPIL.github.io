import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import itertools

def get_score(points):
    """
    Calculate the score: minimum triangle area divided by convex hull area.
    """
    N = len(points)
    if N < 3:
        return 0.0

    # Check for degeneracy
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area < 1e-10:
            return 0.0
    except Exception:
        return 0.0

    # Vectorized triangle area calculation
    # Generate indices for all unique triples
    # For N=13, this is 286 triangles.
    # We can precompute indices for speed in the loop, but get_score is called many times.
    # It's better to keep it simple and fast.
    
    min_area = float('inf')
    # Using loops for N=13 is acceptable in Python if not too slow, 
    # but vectorization is safer for speed.
    
    # Pre-compute indices outside if possible, but here we do it inside.
    # Optimization: Use numpy operations on the whole array.
    # points shape (N, 2)
    
    # Create indices
    # i, j, k such that i < j < k
    # This creates 3 arrays of length M
    # M = N*(N-1)*(N-2)/6
    
    # For N=13, M=286. 
    # Creating index arrays every call is overhead.
    # Let's use a cached index generator or just loops.
    # 286 iterations of simple math is fast.
    
    # Let's stick to loops for clarity and low overhead, or simple numpy if N is larger.
    # Actually, 286 is small.
    
    # To be safe and fast:
    idx = np.indices((N, N, N))
    i, j, k = idx
    mask = (i < j) & (j < k)
    i, j, k = i[mask], j[mask], k[mask]
    
    p_i = points[i]
    p_j = points[j]
    p_k = points[k]
    
    # Cross product magnitude
    # (xj - xi)(yk - yi) - (yj - yi)(xk - xi)
    dx_ji = p_j[:, 0] - p_i[:, 0]
    dy_ji = p_j[:, 1] - p_i[:, 1]
    dx_ki = p_k[:, 0] - p_i[:, 0]
    dy_ki = p_k[:, 1] - p_i[:, 1]
    
    cross = dx_ji * dy_ki - dy_ji * dx_ki
    areas = 0.5 * np.abs(cross)
    min_area = np.min(areas)
    
    return min_area / hull_area

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin
    
    N = 13
    best_score = 0.0
    best_points = None
    
    # 1. Baseline: Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    points = np.column_stack((np.cos(angles), np.sin(angles)))
    best_score = get_score(points)
    best_points = points.copy()
    
    # 2. Optimization
    # We will optimize the 2D coordinates directly.
    # Objective: minimize -get_score
    
    def neg_score(x):
        pts = x.reshape(N, 2)
        return -get_score(pts)
    
    # Initial point: Regular polygon
    x0 = best_points.flatten()
    
    # Run Nelder-Mead
    # Nelder-Mead is derivative-free and handles non-smoothness better than L-BFGS-B
    # for small dimensions (26).
    
    n_restarts = 0
    max_restarts = 20 # Adjust based on time
    
    # We will do a few iterations of:
    # 1. Local opt from current best
    # 2. Local opt from random perturbation of best
    
    while time.time() < deadline:
        n_restarts += 1
        
        if n_restarts == 1:
            x_start = best_points.flatten()
        else:
            # Perturb the best solution
            # Scale of perturbation: 0.1
            noise = np.random.normal(0, 0.1, N * 2)
            x_start = best_points.flatten() + noise
            
        try:
            # Use Nelder-Mead
            # maxiter needs to be high enough to converge
            # xatol, fatol tight
            res = minimize(neg_score, x_start, method='Nelder-Mead',
                           options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
            
            current_score = -res.fun
            if current_score > best_score:
                best_score = current_score
                best_points = res.x.reshape(N, 2).copy()
                
        except Exception as e:
            pass
            
        # Check time
        if time.time() > deadline - 5:
            break

    return best_points
