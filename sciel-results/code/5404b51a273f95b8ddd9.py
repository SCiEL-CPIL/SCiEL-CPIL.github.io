import numpy as np
import time
import itertools
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import random

N = 13

# EVOLVE-BLOCK-START

def _compute_score(points):
    """
    Computes the score: min triangle area / convex hull area.
    points: (N, 2) array
    """
    if len(points) < 3:
        return 0.0
    
    # Compute Convex Hull Area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume # In 2D, volume is area
        if hull_area <= 1e-10:
            return 0.0
    except Exception:
        # Fallback if hull is degenerate (e.g., collinear)
        return 0.0

    # Compute all triangle areas
    # Vectorized calculation
    # Indices of all triples
    # To be fast, we can loop or use broadcasting. 
    # N=13, so 286 triangles. Broadcasting is fine.
    
    # Get all pairs of points
    # Let's just use a loop for clarity and moderate speed, or vectorize.
    # Vectorization:
    # i, j, k indices
    idx = np.arange(N)
    # Create meshgrid for i, j, k
    i_idx, j_idx, k_idx = np.meshgrid(idx, idx, idx, indexing='ij')
    # Filter i < j < k
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_flat = i_idx[mask]
    j_flat = j_idx[mask]
    k_flat = k_idx[mask]
    
    p_i = points[i_flat]
    p_j = points[j_flat]
    p_k = points[k_flat]
    
    # Area = 0.5 * | (p_j - p_i) x (p_k - p_i) |
    # In 2D, cross product is scalar: (x1*y2 - y1*x2)
    v1 = p_j - p_i
    v2 = p_k - p_i
    areas = 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
    
    min_area = np.min(areas)
    
    return min_area / hull_area

def _soft_min_objective(x, tau):
    """
    Objective function for minimization.
    x: flattened array of points (2N,)
    tau: temperature for soft-min
    """
    points = x.reshape((N, 2))
    
    # Compute Hull Area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-10:
            return 1e6
    except Exception:
        return 1e6

    # Compute triangle areas
    idx = np.arange(N)
    i_idx, j_idx, k_idx = np.meshgrid(idx, idx, idx, indexing='ij')
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_flat = i_idx[mask]
    j_flat = j_idx[mask]
    k_flat = k_idx[mask]
    
    p_i = points[i_flat]
    p_j = points[j_flat]
    p_k = points[k_flat]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    areas = 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
    
    # We want to maximize min(areas) / hull_area
    # Equivalent to minimizing - min(areas) / hull_area
    # Or using soft min:
    # f = (1/tau) * log(sum(exp(-area_t / (tau * hull_area))))
    # This is expensive to compute exp for all triangles every iteration.
    # Alternative: Just use the hard min with Nelder-Mead? 
    # Nelder-Mead handles non-smooth functions well.
    # Let's stick to a simpler approach for the objective in minimize:
    # Return -score. But score is non-smooth.
    # Let's use a slightly smoothed version:
    # min_area_smooth = -tau * log(sum(exp(-areas / (tau * hull_area))))
    # We want to maximize this, so minimize -min_area_smooth.
    
    # For speed and stability, let's just use the hard min and rely on Nelder-Mead's robustness.
    # If we use L-BFGS-B, we need gradients. Nelder-Mead is gradient-free.
    
    min_area = np.min(areas)
    score = min_area / hull_area
    return -score

def _local_optimize(points, max_iter=200, tol=1e-8):
    """
    Perform local optimization using Nelder-Mead.
    """
    x0 = points.flatten()
    
    # Nelder-Mead parameters
    options = {
        'maxiter': max_iter,
        'xatol': tol,
        'fatol': tol,
        'disp': False
    }
    
    # We need to be careful with the convex hull calculation inside the objective.
    # It can fail if points become collinear.
    
    try:
        res = minimize(_soft_min_objective, x0, method='Nelder-Mead', options=options)
        if res.success or res.fun < 0: # Check if it found something better
            # Verify the result
            new_points = res.x.reshape((N, 2))
            return new_points
    except Exception:
        pass
        
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Leave 10s margin
    
    best_score = 0.0
    best_points = None
    
    # Initializations
    # 1. Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    radius = 1.0
    init_regular = np.column_stack((radius * np.cos(angles), radius * np.sin(angles)))
    
    # 2. Random points in unit circle
    rng = np.random.RandomState(42)
    init_random = []
    for _ in range(N):
        r = np.sqrt(rng.random())
        theta = rng.random() * 2 * np.pi
        init_random.append([r * np.cos(theta), r * np.sin(theta)])
    init_random = np.array(init_random)
    
    # 3. "Fibonacci" / Vogel spiral projection to 2D? 
    # Or just a grid
    init_grid = []
    # 4x4 grid subset? 13 points.
    # Let's use a hexagonal packing init
    side = 1.0
    for row in range(4):
        for col in range(4):
            if len(init_grid) >= N:
                break
            x = col * side * 0.866
            y = row * side
            if row % 2 == 1:
                x += side * 0.433
            # Center it roughly
            init_grid.append([x - 1.5, y - 1.5])
    init_grid = np.array(init_grid[:N])
    
    inits = [init_regular, init_random, init_grid]
    
    for init_points in inits:
        if time.time() > deadline:
            break
            
        # Try to optimize
        # We might want to try multiple iterations of local optimization
        # or just one big run.
        
        current_points = init_points.copy()
        
        # Step 1: Coarse optimization
        # Nelder-Mead can be sensitive to initial simplex.
        # Let's just run it.
        current_points = _local_optimize(current_points, max_iter=500, tol=1e-6)
        
        # Step 2: Fine optimization
        current_points = _local_optimize(current_points, max_iter=500, tol=1e-9)
        
        score = _compute_score(current_points)
        
        if score > best_score:
            best_score = score
            best_points = current_points.copy()
            
    if best_points is None:
        # Fallback
        best_points = init_regular
        
    return best_points

# EVOLVE-BLOCK-END
